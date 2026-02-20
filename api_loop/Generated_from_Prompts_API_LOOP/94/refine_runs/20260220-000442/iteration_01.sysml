package AutomotiveDomain {
    private import ScalarValues::*;

    // 扭矩是汽车相关的重要物理量，用于标准化表达动力特性
    item def 扭矩 {
        // 数值以标准实数表示，单位可由外部约定为 N·m
        attribute value : Real;
    }

    // 汽车是本模型的核心组件，用于描述汽车领域的基础概念
    part def 汽车 {
        // 关联汽车的扭矩物理量，体现动力特性
        attribute torque : 扭矩;
    }

    // Car 是 “汽车” 的英文别名，便于跨语言交流与使用
    part def Car :> 汽车;
}