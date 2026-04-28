
## File Structure
```
falcon/
├── models/              # MBE 神经元定义（serial + decoder）
│   ├── interfaces.py    # BasisSchedule, MBENeuronConfig
│   ├── mbe_serial.py    # 逐步仿真（参考实现）
│   └── mbe_decoder.py   # 解析事件驱动推理（SRAI）
├── conversion/          # 训练流水线
│   ├── neuron.py        # TrainableMBENeuron（可训练版本）
│   ├── lti.py           # ⭐ LTI 核心实现
│   ├── data.py          # 目标函数 + 数据生成
│   ├── train.py         # 训练循环
│   ├── runner.py        # 统一入口（支持 manual / lti 模式）
│   └── config.py        # YAML 配置解析
├── configs/yaml/        # YAML 配置文件
├── run_comparison.py    # ⭐ 对比实验脚本
└── run_conversion_training.py  # 单次训练脚本
```