package AutomotiveDomain {
    private import ScalarValues::*;

    // 汽车领域标准化定义包
    // 本模型旨在为汽车领域的基本概念和元素提供标准化定义和注释
    // 便于开发者和用户理解每个元素的含义和应用场景

    // 扭矩物理量定义 (Torque physical quantity definition)
    // 用于标准化表达汽车特性中的扭矩相关参数
    part def Torque {
        // 扭矩数值 (Torque value)
        attribute value : Real;
        // 单位说明：牛顿·米 (Unit: Newton-meter)
        attribute unit : String;
    }

    // 汽车核心部件定义 (Automobile core component definition)
    // 别名：汽车 (Car)
    // 作为系统模型的核心组件，涵盖汽车的基本属性和特性
    part def Automobile {
        // 汽车名称属性 (Automobile name attribute)
        attribute name : String;
        // 汽车型号 (Automobile model)
        attribute model : String;
        // 汽车品牌 (Automobile brand)
        attribute brand : String;
        // 最大速度 (Maximum speed) 单位: km/h
        attribute maxSpeed : Real;
        // 发动机功率 (Engine power) 单位: kW
        attribute enginePower : Real;
        // 关联扭矩部件 (Associated torque component)
        // 确保汽车特性的标准化和一致性表达
        part torque : Torque;
    }

    // 汽车实例定义 (Automobile instance definition)
    // 汽车 (Car) - 核心组件实例
    // 为各种应用场景中的通信和使用提供便利
    part Car : Automobile {
        // 默认名称设置 (Default name setting)
        attribute :>> name = "汽车";
        // 默认型号 (Default model)
        attribute :>> model = "标准型";
        // 默认品牌 (Default brand)
        attribute :>> brand = "通用";
        // 默认最大速度 (Default maximum speed)
        attribute :>> maxSpeed = 200.0;
        // 默认发动机功率 (Default engine power)
        attribute :>> enginePower = 150.0;
    }

}