package ScientificConstants {
    private import ScalarValues::*;

    part def MathematicalConstants {
        attribute euler_number : Real = 2.71828182845904523536;
        attribute pi : Real = 3.14159265358979323846;
    }

    part def PhysicalConstants {
        attribute speed_of_light_m_per_s : Real = 299792458.0;
        attribute fine_structure_constant : Real = 0.0072973525693;
        attribute electron_to_proton_mass_ratio : Real = 0.000544617021487;
    }

    part def ContextConstants {
        attribute standard_gravity_m_per_s2 : Real = 9.80665;
        attribute model_x_amplifier_gain : Real = 42.0;
    }

    part def ScientificComputingSystem {
        part mathConsts : MathematicalConstants;
        part physConsts : PhysicalConstants;
        part ctxConsts : ContextConstants;
    }

    part system : ScientificComputingSystem {
        part mathConsts : MathematicalConstants {
            attribute euler_number : Real = 2.71828182845904523536;
            attribute pi : Real = 3.14159265358979323846;
        }
        part physConsts : PhysicalConstants {
            attribute speed_of_light_m_per_s : Real = 299792458.0;
            attribute fine_structure_constant : Real = 0.0072973525693;
            attribute electron_to_proton_mass_ratio : Real = 0.000544617021487;
        }
        part ctxConsts : ContextConstants {
            attribute standard_gravity_m_per_s2 : Real = 9.80665;
            attribute model_x_amplifier_gain : Real = 42.0;
        }
    }
}