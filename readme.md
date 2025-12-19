# 阶段 1: 基础功能测试（快速验证）
pytest -v tests/test_memory_usage.py::test_cpu_memory_budget

pytest -v tests/test_memory_usage.py -m cuda -k "single_gpu_memory_budget"

# 阶段 2: 分布式基础测试（需要 2+ GPU）
pytest -v -s tests/test_memory_usage.py -m distributed -k "multi_gpu_memory_budget"

pytest -v -s tests/test_memory_usage.py -m distributed -k "multi_gpu_communication_memory"

# 阶段 3: Memray 基础测试（生成报告）

pytest -v -s tests/test_memory_usage.py -m memray -k "test_memray_profiling_qubit_circuit"

pytest -v -s tests/test_memory_usage.py -m "memray and cuda" -k "test_memray_profiling_gpu_operations"

# 阶段 4: Memray 规模化测试（可选，耗时较长）
pytest -v -s tests/test_memory_usage.py -m memray -k "test_memray_profiling_cpu_all_scales"

pytest -v -s tests/test_memory_usage.py -m "memray and cuda" -k "test_memray_profiling_gpu_all_scales"

# 阶段 5: Measure 和 Expectation 专项测试（可选）
pytest -v -s tests/test_memory_usage.py -m memray -k "measure"

pytest -v -s tests/test_memory_usage.py -m memray -k "expectation"

# 阶段 6: 通信 memray 测试（需要 2+ GPU）
pytest -v -s tests/test_memory_usage.py -m "memray and distributed" -k "test_memray_profiling_communication"

pytest -v -s tests/test_memory_usage.py -m "memray and distributed" -k "test_memray_profiling_communication_all_scales"

注意事项

需要 CUDA：所有 -m cuda 或 GPU 相关测试需要 CUDA 环境

需要多 GPU：所有 -m distributed 测试需要至少 2 张 GPU

需要 memray：所有 -m memray 测试需要安装 memray（pip install memray）

查看输出：使用 -s 参数可以看到 print 输出和内存监控信息

测试规模：规模化测试默认 4-32 qubits，可在文件开头修改 SCALED_TEST_* 配置

通信测试规模：默认 4-16 qubits（步长 2），可在文件开头修改 COMM_TEST_* 配置

