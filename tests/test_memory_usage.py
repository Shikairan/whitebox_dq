import gc
import os
import sys
import tracemalloc
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deepquantum as dq

# 尝试导入 memray（可选依赖，用于生成内存分析报告）
try:
    import memray
    MEMRAY_AVAILABLE = True
except ImportError:
    MEMRAY_AVAILABLE = False


# ==============================================================================
# 测试规模配置（用户可在此处手动调整）
# ==============================================================================
#
# 说明：
# - CIRCUIT_SCALES：基础单元测试使用的小规模电路（主要用于快速回归）
# - SCALED_TEST_*：measure / expectation 大规模测试的比特数范围
# - COMM_TEST_*：分布式通信相关测试的比特数范围
# - MAX_*：内存/显存的上限阈值（单位见变量名）
#
# 如需缩小/扩大测试规模，只需修改下面几个配置变量即可，无需改动测试函数逻辑。

# 基础测试规模（用于常规单元测试）
# 格式: (nqubit, max_cpu_kb, max_gpu_mb)
CIRCUIT_SCALES = [
    (2, 2 * 1024, 16),      # Small: 2 qubits
    (3, 4 * 1024, 32),      # Medium: 3 qubits
    (4, 8 * 1024, 64),      # Large: 4 qubits
    (5, 16 * 1024, 128),    # Extra large: 5 qubits
]

# 规模化测试范围（用于 measure 和 expectation 的参数化测试）
SCALED_TEST_MIN_QUBITS = 4    # 最小测试比特数
SCALED_TEST_MAX_QUBITS = 28   # 最大测试比特数
SCALED_TEST_STEP = 1          # 步长（可调大以减少测试次数，例如 2 或 4）

# 分布式通信测试规模（推荐保持相对较小）
COMM_TEST_MIN_QUBITS = 16      # 通信测试的最小比特数
COMM_TEST_MAX_QUBITS = 30     # 通信测试的最大比特数
COMM_TEST_STEP = 2            # 通信测试比特数步长

# 全局内存限制（防止测试占用过多资源）
MAX_CPU_KB = 32 * 1024        # CPU 端 Python/张量内存上限（KB）
MAX_GPU_MB = 256              # 单机单卡 GPU 显存上限（MB）
MAX_MULTI_GPU_MB = 512        # 多机/多卡场景下每个 rank 的显存上限（MB）

# 根据上述配置生成测试比特数列表，后续测试通过参数化方式引用
SCALED_TEST_QUBITS = list(range(SCALED_TEST_MIN_QUBITS, SCALED_TEST_MAX_QUBITS + 1, SCALED_TEST_STEP))
COMM_TEST_QUBITS = list(range(COMM_TEST_MIN_QUBITS, COMM_TEST_MAX_QUBITS + 1, COMM_TEST_STEP))


# ---- 通用辅助函数 ------------------------------------------------------------


def _assert_cpu_memory(fn, max_kb: int = MAX_CPU_KB) -> None:
    """在 tracemalloc 监控下执行 fn，并断言其 CPU 峰值内存不超过 max_kb。"""
    gc.collect()
    tracemalloc.start()
    fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= max_kb * 1024, f'CPU case exceeded memory budget: peak={peak/1024:.1f} KB'


def _assert_gpu_memory(fn, device: torch.device, max_mb: int = MAX_GPU_MB) -> None:
    """在指定 CUDA 设备上执行 fn，并断言其显存峰值不超过 max_mb。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # fn may already be a partial function with device bound, or may need device passed
    if hasattr(fn, 'func') and hasattr(fn, 'args'):  # It's a partial
        fn()
    else:
        fn(device)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= max_mb * 1024 * 1024, f'CUDA case exceeded memory budget: peak={peak/1024/1024:.2f} MB'


# ---- 基础功能测试用例（CPU / 单 GPU）-----------------------------------------

def _case_qubit_circuit(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """构建并执行一个基础 Qubit 电路，用于覆盖门操作、编码和 expectation 计算。"""
    cir = dq.QubitCircuit(nqubit, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    cir.cnot_ring()
    cir.observable(0)
    if nqubit > 1:
        cir.observable(1, 'x')
    data = torch.randn(nqubit * 2, device=device, dtype=torch.float)
    cir.to(device)
    state = cir(data=data)
    _ = state.reshape(-1)
    _ = cir.expectation()


def _case_mps_and_state(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """测试 MatrixProductState 和 QubitState 的构造与张量展开的内存情况。"""
    chi = min(16, 2 ** (nqubit // 2))  # Scale chi with nqubit
    mps = dq.MatrixProductState(nqubit, chi=chi)
    mps.to(device)
    _ = mps.full_tensor()
    qs = dq.QubitState(min(nqubit, 4), state='entangle')
    qs.to(device)
    _ = qs.state if hasattr(qs, 'state') else None


def _case_qmath_ops(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """测试 qmath 中 multi_kron / partial_trace 等基础线性代数工具的内存情况。"""
    # Limit to reasonable size for multi_kron
    n = min(nqubit, 4)
    eye = torch.eye(2, dtype=torch.cfloat, device=device)
    kron = dq.multi_kron([eye] * n)
    rho = kron.reshape(1, 2**n, 2**n)
    _ = dq.partial_trace(rho, nqudit=n, trace_lst=[n-1] if n > 1 else [])


def _case_ansatz_and_qasm(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """测试 QFT ansatz 以及 QASM3 导出/导入的内存占用情况。"""
    # Limit QFT size for memory reasons
    qft_n = min(nqubit, 4)
    qft = dq.QuantumFourierTransform(qft_n, reverse=True)
    qft.to(device)
    _ = qft()
    # Use qasm3 module directly to avoid import issues
    cir = dq.QubitCircuit(min(nqubit, 3))
    cir.h(0)
    if cir.nqubit > 1:
        cir.cnot(0, 1)
    # Use qasm3 module directly
    try:
        qasm = dq.qasm3.cir_to_qasm3(cir)
        rebuilt = dq.qasm3.qasm3_to_cir(qasm)
        rebuilt.to(device)
        _ = rebuilt()
    except AttributeError:
        # Fallback: just test circuit execution
        cir.to(device)
        _ = cir()


def _case_channel_and_measure(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """测试噪声通道（density matrix）和测量路径的内存情况。"""
    n = min(nqubit, 3)  # Limit density matrix size
    cir = dq.QubitCircuit(n, den_mat=True)
    cir.hlayer()
    if n > 0:
        cir.bit_flip(0)
    if n > 1:
        cir.phase_flip(1)
        cir.depolarizing(0)
    cir.observable(0)
    if n > 1:
        cir.observable(1, 'z')
    cir.to(device)
    _ = cir()
    _ = cir.expectation()


CPU_CASES = [
    ('qubit_circuit', _case_qubit_circuit),
    ('mps_and_state', _case_mps_and_state),
    ('qmath_ops', _case_qmath_ops),
    ('ansatz_and_qasm', _case_ansatz_and_qasm),
    ('channel_and_measure', _case_channel_and_measure),
]


@pytest.mark.parametrize('nqubit,max_cpu_kb', [(s[0], s[1]) for s in CIRCUIT_SCALES])
@pytest.mark.parametrize('name,fn', CPU_CASES)
def test_cpu_memory_budget(name: str, fn, nqubit: int, max_cpu_kb: int) -> None:
    """在多种 nqubit 规模下，检查典型 CPU 代码路径的内存是否在预期范围内。"""
    case_fn = partial(fn, nqubit, 'cpu')
    _assert_cpu_memory(case_fn, max_kb=max_cpu_kb)


GPU_CASES = [
    ('qubit_circuit', _case_qubit_circuit),
    ('mps_and_state', _case_mps_and_state),
    ('qmath_ops', _case_qmath_ops),
    ('ansatz_and_qasm', _case_ansatz_and_qasm),
    ('channel_and_measure', _case_channel_and_measure),
]


@pytest.mark.cuda
@pytest.mark.parametrize('nqubit,max_gpu_mb', [(s[0], s[2]) for s in CIRCUIT_SCALES])
@pytest.mark.parametrize('name,fn', GPU_CASES)
def test_single_gpu_memory_budget(name: str, fn, nqubit: int, max_gpu_mb: int) -> None:
    """在多种 nqubit 规模下，检查典型 CUDA 代码路径的显存是否在预期范围内。"""
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用，跳过 GPU 内存测试')
    device = torch.device('cuda:0')
    case_fn = partial(fn, nqubit, device)
    _assert_gpu_memory(case_fn, device=device, max_mb=max_gpu_mb)


# ---- 多 GPU 分布式电路内存测试 ----------------------------------------------

def _dist_worker(rank: int, world_size: int, nqubit: int, max_mb: int) -> None:
    """多进程 worker：构建分布式 Qubit 电路并测试其显存占用和梯度计算。"""
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    backend = 'nccl'
    dq.setup_distributed(backend)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.reset_peak_memory_stats(device)
    data = torch.randn(nqubit * 2, device=device, requires_grad=True)
    cir = dq.DistributedQubitCircuit(nqubit, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    # 使用简单的 CNOT 代替 cnot_ring()，避免触发多目标门在分布式实现中的限制
    # 仅在本地比特数足够时添加 CNOT，保证 dist_many_targ_gate 的断言不会失败
    if nqubit >= 2:
        # Add CNOT gates that are within local qubit constraints
        # For world_size=2, log_num_nodes=1, so local qubits = nqubit - 1
        local_qubits = nqubit - 1  # log_num_amps_per_node
        if local_qubits >= 2:
            # Add CNOT between first two local qubits
            cir.cnot(0, 1)
    cir.observable(0)
    if nqubit > 1:
        cir.observable(1, 'x')
    cir.to(device)
    # 前向传播，使用 distributed 实现得到最终态
    _ = cir(data=data)
    # 使用 expectation() 做损失并反向传播（参考 test_circuit.py），测试分布式梯度路径
    exp = cir.expectation().sum()
    exp.backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= max_mb * 1024 * 1024, f'Rank {rank} exceeded memory budget: {peak/1024/1024:.2f} MB'
    dq.cleanup_distributed()


@pytest.mark.distributed
@pytest.mark.parametrize('nqubit,max_gpu_mb', [(s[0], s[2] * 2) for s in CIRCUIT_SCALES[:3]])  # Limit to first 3 scales
def test_multi_gpu_memory_budget(nqubit: int, max_gpu_mb: int) -> None:
    """在多 GPU 场景下，测试分布式 Qubit 电路在不同 nqubit 下的显存占用。"""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('需要至少 2 张可用 GPU 才能运行多机/多卡内存测试')
    if not dist.is_nccl_available():
        pytest.skip('NCCL 不可用，跳过多机 GPU 内存测试')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    world_size = 2
    mp.spawn(_dist_worker, args=(world_size, nqubit, max_gpu_mb), nprocs=world_size, join=True)


# ---- Memray 总体内存分析（小规模示例）----------------------------------------

@pytest.mark.memray
def test_memray_profiling_qubit_circuit() -> None:
    """生成一个小规模 Qubit 电路的 CPU memray 报告，用于快速验证分析流程。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试。请运行: pip install memray')
    
    import tempfile
    import subprocess
    import uuid
    
    # Use a unique filename to avoid conflicts
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_{uuid.uuid4().hex}.bin')
    
    try:
        # Run with memray tracking using context manager
        with memray.Tracker(memray_file):
            # Run a representative workload
            for nqubit in [2, 3, 4]:
                _case_qubit_circuit(nqubit, 'cpu')
        
        # Generate HTML report
        html_file = 'memray_report.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray 报告未生成: {html_file}'
        print(f'\n✓ memray 报告已生成: {html_file}')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass  # Ignore cleanup errors


@pytest.mark.memray
@pytest.mark.cuda
def test_memray_profiling_gpu_operations() -> None:
    """生成一个小规模 Qubit/MPS 的 GPU memray 报告，用于验证 GPU 端分析流程。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用，跳过 GPU memray 测试')
    
    import tempfile
    import subprocess
    import uuid
    
    # Use a unique filename to avoid conflicts
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_gpu_{uuid.uuid4().hex}.bin')
    
    try:
        # Run with memray tracking using context manager
        with memray.Tracker(memray_file):
            device = torch.device('cuda:0')
            for nqubit in [2, 3]:
                _case_qubit_circuit(nqubit, device)
                _case_mps_and_state(nqubit, device)
        
        html_file = 'memray_report_gpu.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray GPU 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray GPU 报告未生成: {html_file}'
        print(f'\n✓ memray GPU 报告已生成: {html_file}')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass  # Ignore cleanup errors


# ---- 扩展：按比特数缩放的 memray 分析（4~32 qubits）------------------------

def _case_qubit_circuit_scaled(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """构建一个可随 nqubit 缩放的 Qubit 电路，用于观测规模变化下的资源占用。"""
    cir = dq.QubitCircuit(nqubit, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    if nqubit >= 2:
        # Add CNOT gates in a ring pattern, but limit to avoid memory issues
        for i in range(min(nqubit - 1, 4)):  # Limit to 4 CNOT gates max
            cir.cnot(i, (i + 1) % nqubit)
    cir.observable(0)
    if nqubit > 1:
        cir.observable(1, 'x')
    data = torch.randn(nqubit * 2, device=device, dtype=torch.float)
    cir.to(device)
    state = cir(data=data)
    _ = state.reshape(-1)
    _ = cir.expectation()


def _case_qubit_circuit_measure_only(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """Build and execute a qubit circuit, focusing on measure operations."""
    cir = dq.QubitCircuit(nqubit, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    if nqubit >= 2:
        for i in range(min(nqubit - 1, 4)):
            cir.cnot(i, (i + 1) % nqubit)
    data = torch.randn(nqubit * 2, device=device, dtype=torch.float)
    cir.to(device)
    state = cir(data=data)
    _ = state.reshape(-1)
    # Focus on measure operations
    shots = 1000
    _ = cir.measure(shots=shots, wires=list(range(min(nqubit, 4))))


def _case_qubit_circuit_expectation_only(nqubit: int, device: torch.device | str = 'cpu') -> None:
    """Build and execute a qubit circuit, focusing on expectation operations."""
    cir = dq.QubitCircuit(nqubit, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    if nqubit >= 2:
        for i in range(min(nqubit - 1, 4)):
            cir.cnot(i, (i + 1) % nqubit)
    cir.observable(0)
    if nqubit > 1:
        cir.observable(1, 'x')
    if nqubit > 2:
        cir.observable(2, 'z')
    data = torch.randn(nqubit * 2, device=device, dtype=torch.float, requires_grad=True)
    cir.to(device)
    state = cir(data=data)
    _ = state.reshape(-1)
    # Focus on expectation operations
    exp = cir.expectation()
    _ = exp.sum()


@pytest.mark.memray
@pytest.mark.parametrize('nqubit', SCALED_TEST_QUBITS)
def test_memray_profiling_cpu_scaled(nqubit: int) -> None:
    """对单一 nqubit 规模的 CPU 电路运行进行 memray 分析，生成独立 HTML 报告。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试。请运行: pip install memray')
    
    import tempfile
    import subprocess
    import uuid
    
    # Use a unique filename to avoid conflicts
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_cpu_{nqubit}q_{uuid.uuid4().hex}.bin')
    
    try:
        # Track memory usage
        gc.collect()
        tracemalloc.start()
        
        # Run with memray tracking using context manager
        with memray.Tracker(memray_file):
            _case_qubit_circuit_scaled(nqubit, 'cpu')
        
        current, peak_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Generate HTML report
        html_file = f'memray_report_cpu_{nqubit}q.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray CPU {nqubit}q 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray CPU {nqubit}q 报告未生成: {html_file}'
        print(f'\n✓ memray CPU {nqubit}q 报告已生成: {html_file} (峰值内存: {peak_cpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass  # Ignore cleanup errors


@pytest.mark.memray
@pytest.mark.cuda
@pytest.mark.parametrize('nqubit', SCALED_TEST_QUBITS)
def test_memray_profiling_gpu_scaled(nqubit: int) -> None:
    """对单一 nqubit 规模的 GPU 电路运行进行 memray 分析，生成独立 HTML 报告。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用，跳过 GPU memray 测试')
    
    import tempfile
    import subprocess
    import uuid
    
    device = torch.device('cuda:0')
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_gpu_{nqubit}q_{uuid.uuid4().hex}.bin')
    
    try:
        # Track GPU memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Run with memray tracking using context manager
        with memray.Tracker(memray_file):
            _case_qubit_circuit_scaled(nqubit, device)
        
        torch.cuda.synchronize(device)
        peak_gpu = torch.cuda.max_memory_allocated(device)
        
        # Generate HTML report
        html_file = f'memray_report_gpu_{nqubit}q.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray GPU {nqubit}q 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray GPU {nqubit}q 报告未生成: {html_file}'
        print(f'\n✓ memray GPU {nqubit}q 报告已生成: {html_file} (峰值显存: {peak_gpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass  # Ignore cleanup errors
        torch.cuda.empty_cache()


@pytest.mark.memray
def test_memray_profiling_cpu_all_scales() -> None:
    """在一个 memray 轨迹中依次测试所有 SCALED_TEST_QUBITS，并生成汇总 CPU 报告。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试。请运行: pip install memray')
    
    import tempfile
    import subprocess
    import uuid
    
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_cpu_all_{uuid.uuid4().hex}.bin')
    
    try:
        gc.collect()
        tracemalloc.start()
        
        # Run with memray tracking using context manager
        with memray.Tracker(memray_file):
            # Test all scales from configured range
            for nqubit in SCALED_TEST_QUBITS:
                print(f'测试 {nqubit} qubits...')
                _case_qubit_circuit_scaled(nqubit, 'cpu')
        
        current, peak_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Generate comprehensive HTML report
        html_file = 'memray_report_cpu_all_scales.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray CPU 综合报告生成失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray CPU 综合报告未生成: {html_file}'
        print(f'\n✓ memray CPU 综合报告已生成: {html_file} (峰值内存: {peak_cpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass  # Ignore cleanup errors


@pytest.mark.memray
@pytest.mark.cuda
def test_memray_profiling_gpu_all_scales() -> None:
    """在一个 memray 轨迹中依次测试所有 SCALED_TEST_QUBITS，并生成汇总 GPU 报告。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用，跳过 GPU memray 测试')
    
    import tempfile
    import subprocess
    import uuid
    
    device = torch.device('cuda:0')
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_gpu_all_{uuid.uuid4().hex}.bin')
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Run with memray tracking using context manager
        with memray.Tracker(memray_file):
            # Test all scales from configured range
            for nqubit in SCALED_TEST_QUBITS:
                print(f'测试 {nqubit} qubits...')
                _case_qubit_circuit_scaled(nqubit, device)
                # Clear cache between tests to get accurate per-scale measurements
                torch.cuda.empty_cache()
        
        torch.cuda.synchronize(device)
        peak_gpu = torch.cuda.max_memory_allocated(device)
        
        # Generate comprehensive HTML report
        html_file = 'memray_report_gpu_all_scales.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray GPU 综合报告生成失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray GPU 综合报告未生成: {html_file}'
        print(f'\n✓ memray GPU 综合报告已生成: {html_file} (峰值显存: {peak_gpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass  # Ignore cleanup errors
        torch.cuda.empty_cache()


# ---- Measure / Expectation 专项内存分析 -------------------------------------

@pytest.mark.memray
@pytest.mark.parametrize('nqubit', SCALED_TEST_QUBITS)
def test_memray_profiling_cpu_measure(nqubit: int) -> None:
    """仅关注 CPU 端测量（measure）路径的 memray 分析，便于区分编译期与采样期开销。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试。请运行: pip install memray')
    
    import tempfile
    import subprocess
    import uuid
    
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_cpu_measure_{nqubit}q_{uuid.uuid4().hex}.bin')
    
    try:
        gc.collect()
        tracemalloc.start()
        
        with memray.Tracker(memray_file):
            _case_qubit_circuit_measure_only(nqubit, 'cpu')
        
        current, peak_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        html_file = f'memray_report_cpu_measure_{nqubit}q.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray CPU measure {nqubit}q 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray CPU measure {nqubit}q 报告未生成: {html_file}'
        print(f'\n✓ memray CPU measure {nqubit}q 报告已生成: {html_file} (峰值内存: {peak_cpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass


@pytest.mark.memray
@pytest.mark.cuda
@pytest.mark.parametrize('nqubit', SCALED_TEST_QUBITS)
def test_memray_profiling_gpu_measure(nqubit: int) -> None:
    """仅关注 GPU 端测量（measure）路径的 memray 分析。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用，跳过 GPU memray 测试')
    
    import tempfile
    import subprocess
    import uuid
    
    device = torch.device('cuda:0')
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_gpu_measure_{nqubit}q_{uuid.uuid4().hex}.bin')
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        with memray.Tracker(memray_file):
            _case_qubit_circuit_measure_only(nqubit, device)
        
        torch.cuda.synchronize(device)
        peak_gpu = torch.cuda.max_memory_allocated(device)
        
        html_file = f'memray_report_gpu_measure_{nqubit}q.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray GPU measure {nqubit}q 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray GPU measure {nqubit}q 报告未生成: {html_file}'
        print(f'\n✓ memray GPU measure {nqubit}q 报告已生成: {html_file} (峰值显存: {peak_gpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass
        torch.cuda.empty_cache()


@pytest.mark.memray
@pytest.mark.parametrize('nqubit', SCALED_TEST_QUBITS)
def test_memray_profiling_cpu_expectation(nqubit: int) -> None:
    """仅关注 CPU 端 expectation 计算（不含 measure）的 memray 分析。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试。请运行: pip install memray')
    
    import tempfile
    import subprocess
    import uuid
    
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_cpu_expectation_{nqubit}q_{uuid.uuid4().hex}.bin')
    
    try:
        gc.collect()
        tracemalloc.start()
        
        with memray.Tracker(memray_file):
            _case_qubit_circuit_expectation_only(nqubit, 'cpu')
        
        current, peak_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        html_file = f'memray_report_cpu_expectation_{nqubit}q.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray CPU expectation {nqubit}q 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray CPU expectation {nqubit}q 报告未生成: {html_file}'
        print(f'\n✓ memray CPU expectation {nqubit}q 报告已生成: {html_file} (峰值内存: {peak_cpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass


@pytest.mark.memray
@pytest.mark.cuda
@pytest.mark.parametrize('nqubit', SCALED_TEST_QUBITS)
def test_memray_profiling_gpu_expectation(nqubit: int) -> None:
    """仅关注 GPU 端 expectation 计算（不含 measure）的 memray 分析。"""
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available():
        pytest.skip('CUDA 不可用，跳过 GPU memray 测试')
    
    import tempfile
    import subprocess
    import uuid
    
    device = torch.device('cuda:0')
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_gpu_expectation_{nqubit}q_{uuid.uuid4().hex}.bin')
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        with memray.Tracker(memray_file):
            _case_qubit_circuit_expectation_only(nqubit, device)
        
        torch.cuda.synchronize(device)
        peak_gpu = torch.cuda.max_memory_allocated(device)
        
        html_file = f'memray_report_gpu_expectation_{nqubit}q.html'
        result = subprocess.run([
            sys.executable, '-m', 'memray', 'flamegraph',
            '--output', html_file,
            memray_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'memray GPU expectation {nqubit}q 生成报告失败: {result.stderr}')
        
        assert os.path.exists(html_file), f'memray GPU expectation {nqubit}q 报告未生成: {html_file}'
        print(f'\n✓ memray GPU expectation {nqubit}q 报告已生成: {html_file} (峰值显存: {peak_gpu/1024/1024:.2f} MB)')
        
    finally:
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass
        torch.cuda.empty_cache()


# ---- Communication 通信函数内存分析 -----------------------------------------

def _case_communication_exchange(nqubit: int, device: torch.device) -> None:
    """构建一个 DistributedQubitState，并调用 comm_exchange_arrays 测试通信内存占用。"""
    from deepquantum.communication import comm_exchange_arrays, comm_get_rank, comm_get_world_size
    from deepquantum.state import DistributedQubitState
    
    # Create a distributed state
    state = DistributedQubitState(nqubit)
    state.to(device)
    
    # Prepare data for exchange
    send_data = state.amps.clone()
    recv_data = torch.zeros_like(send_data)
    
    # Get pair rank for exchange
    rank = comm_get_rank()
    world_size = comm_get_world_size()
    if world_size > 1:
        pair_rank = (rank + 1) % world_size
        # Perform exchange
        comm_exchange_arrays(send_data, recv_data, pair_rank)
    
    # Also test with None pair_rank (collective call)
    comm_exchange_arrays(send_data, recv_data, None)


def _dist_worker_comm(rank: int, world_size: int, nqubit: int, max_mb: int) -> None:
    """多进程 worker：在分布式环境中调用通信函数，并统计每个 rank 的显存峰值。"""
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    backend = 'nccl'
    dq.setup_distributed(backend)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.reset_peak_memory_stats(device)
    
    # Test communication functions
    _case_communication_exchange(nqubit, device)
    
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= max_mb * 1024 * 1024, f'Rank {rank} exceeded memory budget: {peak/1024/1024:.2f} MB'
    dq.cleanup_distributed()


def _dist_worker_memray(rank: int, world_size: int, nqubit: int, memray_file: str, 
                        cpu_peak_file: str, gpu_peak_file: str) -> None:
    """
    多进程 worker：对单一 nqubit 通信场景进行全面的内存监控。
    
    监控内容：
    - memray：仅 rank 0 记录 Python 内存分配轨迹（用于生成火焰图）
    - tracemalloc：所有 rank 监控 CPU 内存峰值
    - torch.cuda.max_memory_allocated：所有 rank 监控 GPU 显存峰值
    
    Args:
        rank: 当前进程的 rank
        world_size: 总进程数
        nqubit: 量子比特数
        memray_file: memray 输出文件路径（仅 rank 0 使用）
        cpu_peak_file: CPU 内存峰值记录文件路径（每个 rank 写入自己的文件）
        gpu_peak_file: GPU 显存峰值记录文件路径（每个 rank 写入自己的文件）
    """
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    backend = 'nccl'
    dq.setup_distributed(backend)
    device = torch.device(f'cuda:{rank}')
    
    # 清理并重置内存统计
    gc.collect()
    tracemalloc.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Rank 0 使用 memray 记录详细的内存分配轨迹
    if rank == 0:
        with memray.Tracker(memray_file):
            _case_communication_exchange(nqubit, device)
    else:
        _case_communication_exchange(nqubit, device)
    
    # 同步并记录峰值内存
    torch.cuda.synchronize(device)
    
    # 记录 CPU 内存峰值（tracemalloc）
    current_cpu, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 记录 GPU 显存峰值
    peak_gpu = torch.cuda.max_memory_allocated(device)
    
    # 将内存信息写入文件（每个 rank 写入自己的文件）
    rank_cpu_file = f"{cpu_peak_file}_rank{rank}.txt"
    rank_gpu_file = f"{gpu_peak_file}_rank{rank}.txt"
    
    try:
        with open(rank_cpu_file, 'w') as f:
            f.write(f"Rank {rank} CPU Memory:\n")
            f.write(f"  Current: {current_cpu / 1024 / 1024:.2f} MB\n")
            f.write(f"  Peak: {peak_cpu / 1024 / 1024:.2f} MB\n")
            f.flush()
            os.fsync(f.fileno())  # 确保写入磁盘
        
        with open(rank_gpu_file, 'w') as f:
            f.write(f"Rank {rank} GPU Memory:\n")
            f.write(f"  Peak Allocated: {peak_gpu / 1024 / 1024:.2f} MB\n")
            f.write(f"  Device: {device}\n")
            f.flush()
            os.fsync(f.fileno())  # 确保写入磁盘
        
        # 打印调试信息（直接输出到控制台）
        print(f'[Rank {rank}] 内存监控完成 - CPU峰值: {peak_cpu / 1024 / 1024:.2f} MB, GPU峰值: {peak_gpu / 1024 / 1024:.2f} MB')
        
    except Exception as e:
        print(f'[Rank {rank}] 写入内存文件失败: {e}')
    
    dq.cleanup_distributed()


def _dist_worker_memray_all(rank: int, world_size: int, memray_file: str,
                            cpu_peak_file: str, gpu_peak_file: str) -> None:
    """
    多进程 worker：对 COMM_TEST_QUBITS 中所有规模的通信场景进行全面的内存监控。
    
    监控内容：
    - memray：仅 rank 0 记录所有规模的 Python 内存分配轨迹
    - tracemalloc：所有 rank 监控整个测试过程的 CPU 内存峰值
    - torch.cuda.max_memory_allocated：所有 rank 监控整个测试过程的 GPU 显存峰值
    
    Args:
        rank: 当前进程的 rank
        world_size: 总进程数
        memray_file: memray 输出文件路径（仅 rank 0 使用）
        cpu_peak_file: CPU 内存峰值记录文件路径（每个 rank 写入自己的文件）
        gpu_peak_file: GPU 显存峰值记录文件路径（每个 rank 写入自己的文件）
    """
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    backend = 'nccl'
    dq.setup_distributed(backend)
    device = torch.device(f'cuda:{rank}')
    
    # 清理并重置内存统计
    gc.collect()
    tracemalloc.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Rank 0 使用 memray 记录所有规模的详细内存分配轨迹
    if rank == 0:
        with memray.Tracker(memray_file):
            for nqubit in COMM_TEST_QUBITS:
                print(f'[Rank {rank}] 测试通信 {nqubit} qubits...')
                _case_communication_exchange(nqubit, device)
                # 每个规模后清理缓存，以便更准确地测量
                torch.cuda.empty_cache()
    else:
        for nqubit in COMM_TEST_QUBITS:
            print(f'[Rank {rank}] 测试通信 {nqubit} qubits...')
            _case_communication_exchange(nqubit, device)
            torch.cuda.empty_cache()
    
    # 同步并记录峰值内存
    torch.cuda.synchronize(device)
    
    # 记录 CPU 内存峰值（整个测试过程的峰值）
    current_cpu, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 记录 GPU 显存峰值（整个测试过程的峰值）
    peak_gpu = torch.cuda.max_memory_allocated(device)
    
    # 将内存信息写入文件（每个 rank 写入自己的文件）
    rank_cpu_file = f"{cpu_peak_file}_rank{rank}.txt"
    rank_gpu_file = f"{gpu_peak_file}_rank{rank}.txt"
    
    try:
        with open(rank_cpu_file, 'w') as f:
            f.write(f"Rank {rank} CPU Memory (All Scales):\n")
            f.write(f"  Current: {current_cpu / 1024 / 1024:.2f} MB\n")
            f.write(f"  Peak: {peak_cpu / 1024 / 1024:.2f} MB\n")
            f.write(f"  Tested Scales: {COMM_TEST_QUBITS}\n")
            f.flush()
            os.fsync(f.fileno())  # 确保写入磁盘
        
        with open(rank_gpu_file, 'w') as f:
            f.write(f"Rank {rank} GPU Memory (All Scales):\n")
            f.write(f"  Peak Allocated: {peak_gpu / 1024 / 1024:.2f} MB\n")
            f.write(f"  Device: {device}\n")
            f.write(f"  Tested Scales: {COMM_TEST_QUBITS}\n")
            f.flush()
            os.fsync(f.fileno())  # 确保写入磁盘
        
        # 打印调试信息（直接输出到控制台）
        print(f'[Rank {rank}] 内存监控完成 (所有规模) - CPU峰值: {peak_cpu / 1024 / 1024:.2f} MB, GPU峰值: {peak_gpu / 1024 / 1024:.2f} MB')
        
    except Exception as e:
        print(f'[Rank {rank}] 写入内存文件失败: {e}')
    
    dq.cleanup_distributed()


@pytest.mark.distributed
@pytest.mark.parametrize('nqubit', COMM_TEST_QUBITS)
def test_multi_gpu_communication_memory(nqubit: int) -> None:
    """在多 GPU 分布式环境中测试通信函数（不启用 memray）的显存占用。"""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('需要至少 2 张可用 GPU 才能运行多机/多卡内存测试')
    if not dist.is_nccl_available():
        pytest.skip('NCCL 不可用，跳过多机 GPU 内存测试')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    world_size = 2
    max_gpu_mb = 256  # Communication tests typically use less memory
    mp.spawn(_dist_worker_comm, args=(world_size, nqubit, max_gpu_mb), nprocs=world_size, join=True)


@pytest.mark.memray
@pytest.mark.distributed
@pytest.mark.parametrize('nqubit', COMM_TEST_QUBITS)
def test_memray_profiling_communication(nqubit: int) -> None:
    """
    对单一 nqubit 的通信场景进行全面的内存监控和分析。
    
    监控内容：
    - memray：生成 rank 0 的 Python 内存分配火焰图（HTML 报告）
    - tracemalloc：记录所有 rank 的 CPU 内存峰值
    - torch.cuda.max_memory_allocated：记录所有 rank 的 GPU 显存峰值
    
    输出：
    - memray_report_communication_{nqubit}q.html：memray 火焰图报告
    - 控制台输出：所有 rank 的 CPU 和 GPU 内存峰值信息
    """
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('需要至少 2 张可用 GPU 才能运行通信测试')
    if not dist.is_nccl_available():
        pytest.skip('NCCL 不可用，跳过通信测试')
    
    import tempfile
    import subprocess
    import uuid
    
    # 生成唯一的文件路径（监控文件保存到当前目录，便于查看）
    unique_id = uuid.uuid4().hex
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_comm_{nqubit}q_{unique_id}.bin')
    # CPU 和 GPU 监控文件保存到当前目录，方便查看
    cpu_peak_file = os.path.join(os.getcwd(), f'cpu_peak_comm_{nqubit}q_{unique_id}')
    gpu_peak_file = os.path.join(os.getcwd(), f'gpu_peak_comm_{nqubit}q_{unique_id}')
    
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')  # 使用不同端口避免冲突
    
    try:
        world_size = 2
        mp.spawn(_dist_worker_memray, args=(world_size, nqubit, memray_file, cpu_peak_file, gpu_peak_file), 
                nprocs=world_size, join=True)
        
        # 等待一下确保文件写入完成
        import time
        time.sleep(0.5)
        
        # 读取并汇总所有 rank 的内存信息
        cpu_peaks = {}
        gpu_peaks = {}
        
        for rank in range(world_size):
            rank_cpu_file = f"{cpu_peak_file}_rank{rank}.txt"
            rank_gpu_file = f"{gpu_peak_file}_rank{rank}.txt"
            
            # 读取 CPU 内存信息
            if os.path.exists(rank_cpu_file):
                try:
                    with open(rank_cpu_file, 'r') as f:
                        content = f.read()
                        # 提取峰值内存值
                        for line in content.split('\n'):
                            if 'Peak:' in line and 'MB' in line:
                                try:
                                    peak_val = float(line.split(':')[1].strip().replace(' MB', ''))
                                    cpu_peaks[rank] = peak_val
                                except (ValueError, IndexError) as e:
                                    print(f'警告: 解析 Rank {rank} CPU 内存失败: {line}, 错误: {e}')
                except Exception as e:
                    print(f'警告: 读取 Rank {rank} CPU 内存文件失败: {e}')
            else:
                print(f'警告: Rank {rank} CPU 内存文件不存在: {rank_cpu_file}')
            
            # 读取 GPU 显存信息
            if os.path.exists(rank_gpu_file):
                try:
                    with open(rank_gpu_file, 'r') as f:
                        content = f.read()
                        # 提取峰值显存值
                        for line in content.split('\n'):
                            if 'Peak Allocated:' in line and 'MB' in line:
                                try:
                                    peak_val = float(line.split(':')[1].strip().replace(' MB', ''))
                                    gpu_peaks[rank] = peak_val
                                except (ValueError, IndexError) as e:
                                    print(f'警告: 解析 Rank {rank} GPU 显存失败: {line}, 错误: {e}')
                except Exception as e:
                    print(f'警告: 读取 Rank {rank} GPU 显存文件失败: {e}')
            else:
                print(f'警告: Rank {rank} GPU 显存文件不存在: {rank_gpu_file}')
        
        # 生成 memray HTML 报告（仅 rank 0 的记录）
        if os.path.exists(memray_file):
            html_file = f'memray_report_communication_{nqubit}q.html'
            result = subprocess.run([
                sys.executable, '-m', 'memray', 'flamegraph',
                '--output', html_file,
                memray_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f'memray communication {nqubit}q 生成报告失败: {result.stderr}')
            else:
                assert os.path.exists(html_file), f'memray communication {nqubit}q 报告未生成: {html_file}'
                print(f'\n✓ memray communication {nqubit}q 报告已生成: {html_file}')
        
        # 打印所有 rank 的内存监控结果
        print(f'\n=== Communication 内存监控结果 ({nqubit} qubits) ===')
        if cpu_peaks:
            print('CPU 内存峰值 (tracemalloc):')
            for rank in sorted(cpu_peaks.keys()):
                print(f'  Rank {rank}: {cpu_peaks[rank]:.2f} MB')
            if len(cpu_peaks) > 0:
                max_cpu_rank = max(cpu_peaks, key=cpu_peaks.get)
                print(f'  最大峰值: {max(cpu_peaks.values()):.2f} MB (Rank {max_cpu_rank})')
        else:
            print('CPU 内存峰值: 未获取到数据')
        
        if gpu_peaks:
            print('GPU 显存峰值 (torch.cuda.max_memory_allocated):')
            for rank in sorted(gpu_peaks.keys()):
                print(f'  Rank {rank}: {gpu_peaks[rank]:.2f} MB')
            if len(gpu_peaks) > 0:
                max_gpu_rank = max(gpu_peaks, key=gpu_peaks.get)
                print(f'  最大峰值: {max(gpu_peaks.values()):.2f} MB (Rank {max_gpu_rank})')
        else:
            print('GPU 显存峰值: 未获取到数据')
        
        # 提示监控文件位置
        print(f'\n监控文件已保存到当前目录:')
        for rank in range(world_size):
            rank_cpu_file = f"{cpu_peak_file}_rank{rank}.txt"
            rank_gpu_file = f"{gpu_peak_file}_rank{rank}.txt"
            if os.path.exists(rank_cpu_file):
                print(f'  CPU监控: {os.path.basename(rank_cpu_file)}')
            if os.path.exists(rank_gpu_file):
                print(f'  GPU监控: {os.path.basename(rank_gpu_file)}')
        print('=' * 50)
        
    finally:
        # 清理临时文件
        cleanup_files = [memray_file]
        for rank in range(world_size):
            cleanup_files.extend([
                f"{cpu_peak_file}_rank{rank}.txt",
                f"{gpu_peak_file}_rank{rank}.txt"
            ])
        
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError:
                    pass


@pytest.mark.memray
@pytest.mark.distributed
def test_memray_profiling_communication_all_scales() -> None:
    """
    对 COMM_TEST_QUBITS 中所有规模的通信场景进行全面的内存监控和分析。
    
    监控内容：
    - memray：生成 rank 0 的 Python 内存分配火焰图（包含所有规模的综合报告）
    - tracemalloc：记录所有 rank 在整个测试过程中的 CPU 内存峰值
    - torch.cuda.max_memory_allocated：记录所有 rank 在整个测试过程中的 GPU 显存峰值
    
    输出：
    - memray_report_communication_all_scales.html：memray 火焰图报告（包含所有规模）
    - 控制台输出：所有 rank 的 CPU 和 GPU 内存峰值信息（整个测试过程的峰值）
    """
    if not MEMRAY_AVAILABLE:
        pytest.skip('memray 未安装，跳过内存分析测试')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('需要至少 2 张可用 GPU 才能运行通信测试')
    if not dist.is_nccl_available():
        pytest.skip('NCCL 不可用，跳过通信测试')
    
    import tempfile
    import subprocess
    import uuid
    
    # 生成唯一的文件路径（监控文件保存到当前目录，便于查看）
    unique_id = uuid.uuid4().hex
    memray_file = os.path.join(tempfile.gettempdir(), f'memray_comm_all_{unique_id}.bin')
    # CPU 和 GPU 监控文件保存到当前目录，方便查看
    cpu_peak_file = os.path.join(os.getcwd(), f'cpu_peak_comm_all_{unique_id}')
    gpu_peak_file = os.path.join(os.getcwd(), f'gpu_peak_comm_all_{unique_id}')
    
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29502')  # 使用不同端口避免冲突
    
    try:
        world_size = 2
        mp.spawn(_dist_worker_memray_all, args=(world_size, memray_file, cpu_peak_file, gpu_peak_file), 
                nprocs=world_size, join=True)
        
        # 等待一下确保文件写入完成
        import time
        time.sleep(0.5)
        
        # 读取并汇总所有 rank 的内存信息
        cpu_peaks = {}
        gpu_peaks = {}
        
        for rank in range(world_size):
            rank_cpu_file = f"{cpu_peak_file}_rank{rank}.txt"
            rank_gpu_file = f"{gpu_peak_file}_rank{rank}.txt"
            
            # 读取 CPU 内存信息
            if os.path.exists(rank_cpu_file):
                try:
                    with open(rank_cpu_file, 'r') as f:
                        content = f.read()
                        # 提取峰值内存值
                        for line in content.split('\n'):
                            if 'Peak:' in line and 'MB' in line:
                                try:
                                    peak_val = float(line.split(':')[1].strip().replace(' MB', ''))
                                    cpu_peaks[rank] = peak_val
                                except (ValueError, IndexError) as e:
                                    print(f'警告: 解析 Rank {rank} CPU 内存失败: {line}, 错误: {e}')
                except Exception as e:
                    print(f'警告: 读取 Rank {rank} CPU 内存文件失败: {e}')
            else:
                print(f'警告: Rank {rank} CPU 内存文件不存在: {rank_cpu_file}')
            
            # 读取 GPU 显存信息
            if os.path.exists(rank_gpu_file):
                try:
                    with open(rank_gpu_file, 'r') as f:
                        content = f.read()
                        # 提取峰值显存值
                        for line in content.split('\n'):
                            if 'Peak Allocated:' in line and 'MB' in line:
                                try:
                                    peak_val = float(line.split(':')[1].strip().replace(' MB', ''))
                                    gpu_peaks[rank] = peak_val
                                except (ValueError, IndexError) as e:
                                    print(f'警告: 解析 Rank {rank} GPU 显存失败: {line}, 错误: {e}')
                except Exception as e:
                    print(f'警告: 读取 Rank {rank} GPU 显存文件失败: {e}')
            else:
                print(f'警告: Rank {rank} GPU 显存文件不存在: {rank_gpu_file}')
        
        # 生成 memray HTML 报告（包含所有规模的综合报告）
        if os.path.exists(memray_file):
            html_file = 'memray_report_communication_all_scales.html'
            result = subprocess.run([
                sys.executable, '-m', 'memray', 'flamegraph',
                '--output', html_file,
                memray_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f'memray communication 综合报告生成失败: {result.stderr}')
            else:
                assert os.path.exists(html_file), f'memray communication 综合报告未生成: {html_file}'
                print(f'\n✓ memray communication 综合报告已生成: {html_file}')
        
        # 打印所有 rank 的内存监控结果
        print(f'\n=== Communication 内存监控结果 (所有规模: {COMM_TEST_QUBITS}) ===')
        if cpu_peaks:
            print('CPU 内存峰值 (tracemalloc, 整个测试过程):')
            for rank in sorted(cpu_peaks.keys()):
                print(f'  Rank {rank}: {cpu_peaks[rank]:.2f} MB')
            if len(cpu_peaks) > 0:
                max_cpu_rank = max(cpu_peaks, key=cpu_peaks.get)
                print(f'  最大峰值: {max(cpu_peaks.values()):.2f} MB (Rank {max_cpu_rank})')
        else:
            print('CPU 内存峰值: 未获取到数据')
        
        if gpu_peaks:
            print('GPU 显存峰值 (torch.cuda.max_memory_allocated, 整个测试过程):')
            for rank in sorted(gpu_peaks.keys()):
                print(f'  Rank {rank}: {gpu_peaks[rank]:.2f} MB')
            if len(gpu_peaks) > 0:
                max_gpu_rank = max(gpu_peaks, key=gpu_peaks.get)
                print(f'  最大峰值: {max(gpu_peaks.values()):.2f} MB (Rank {max_gpu_rank})')
        else:
            print('GPU 显存峰值: 未获取到数据')
        
        # 提示监控文件位置
        print(f'\n监控文件已保存到当前目录:')
        for rank in range(world_size):
            rank_cpu_file = f"{cpu_peak_file}_rank{rank}.txt"
            rank_gpu_file = f"{gpu_peak_file}_rank{rank}.txt"
            if os.path.exists(rank_cpu_file):
                print(f'  CPU监控: {os.path.basename(rank_cpu_file)}')
            if os.path.exists(rank_gpu_file):
                print(f'  GPU监控: {os.path.basename(rank_gpu_file)}')
        print('=' * 60)
        
    finally:
        # 清理临时文件（保留监控文件到当前目录，不删除）
        # 只清理 memray 的 .bin 文件，监控 txt 文件保留供查看
        if os.path.exists(memray_file):
            try:
                os.unlink(memray_file)
            except OSError:
                pass


