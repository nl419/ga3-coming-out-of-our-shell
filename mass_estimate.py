import numpy as np

# Material properties
rho_copper_tube = 0.2   # kg / m
rho_acrylic_pipe = 0.65 # kg / m
m_nozzle = 0.025        # kg
rho_abs_sheet = 2.39    # kg / m2
rho_3D_printed = 1150   # kg / m3
m_big_o_ring = 5.3e-3   # kg
m_small_o_ring = 0.8e-3 # kg   

mass_limit = 1.05
tube_length_limit_one_tube_pass = 0.35 - 40e-3 * 2 - 8e-3 * 2 + 5.5e-3 * 2
tube_length_limit_even_tube_pass = 0.35 - 40e-3 - 20e-3 - 8e-3 * 2 + 5.5e-3 * 2

# Calculate mass of unchanging parts
def _constant_mass(num_shell_passes, num_tube_passes, num_tubes, num_baffles):
    result = 0

    # End plates
    V_endplate = 18592.862e-9 # m^3
    result += 2 * rho_3D_printed * V_endplate

    # Tubesheet
    V_tubesheet = 26080.357e-9 # m^3
    V_hole = 565.047e-9 # m^3
    
    result += 2 * rho_3D_printed * (V_tubesheet - num_tubes * V_hole)

    # Length of copper stuck in the tubesheet
    result += 2 * 8e-3 * num_tubes * rho_copper_tube

    # O-rings
    result += 6 * m_big_o_ring
    result += 2 * num_tubes * m_small_o_ring

    # Nozzles
    result += 4 * m_nozzle

    # Baffles
    baffle_area_fraction = 2382.797 / 3186.902
    A_baffle = 3186.902e-6 # m^2
    A_hole = np.pi * 4e-3 ** 2
    result += num_baffles * rho_abs_sheet * (A_baffle - num_tubes * A_hole) * baffle_area_fraction

    # Header separator
    half_width_separator = 63.7e-3 / 2
    if num_tube_passes == 1:
        num_separator_halves_front = 0
        num_separator_halves_back = 0
    elif num_tube_passes == 2:
        num_separator_halves_front = 2
        num_separator_halves_back = 0
    elif num_tube_passes == 3:
        num_separator_halves_front = 2
        num_separator_halves_back = 2
    else:
        # Assume num_tube_passes == 4. Can't be bothered to implement more.
        num_separator_halves_front = 3
        num_separator_halves_back = 2

    # Front header
    l_small_header = 20e-3
    l_large_header = 40e-3
    result += l_large_header * (rho_acrylic_pipe + num_separator_halves_front * half_width_separator * rho_abs_sheet)
    # Rear header
    if num_tube_passes % 2 == 1:
        result += l_large_header * (rho_acrylic_pipe + num_separator_halves_back * half_width_separator * rho_abs_sheet)
    else:
        result += l_small_header * (rho_acrylic_pipe + num_separator_halves_back * half_width_separator * rho_abs_sheet)

    # Note: the fixed mass is the same, regardless of num_shell_passes
    return result

def calculate_tube_length_baffle_spacing(num_shell_passes, num_tube_passes, num_tubes, num_baffles):
    """Find the maximum length of tubes and baffle spacing for a given geometry configuration"""
    mass_remainder = mass_limit - _constant_mass(num_shell_passes, num_tube_passes, num_tubes, num_baffles)
    rho_shell_tubes = rho_copper_tube * num_tubes + rho_acrylic_pipe
    L_shell_header = 40e-3

    if num_tube_passes % 2 == 1:
        tube_length_limit = tube_length_limit_one_tube_pass
    else:
        tube_length_limit = tube_length_limit_even_tube_pass

    if num_shell_passes == 1:
        tube_length = mass_remainder / rho_shell_tubes
        tube_length = min(tube_length, tube_length_limit)
        baffle_length = tube_length - L_shell_header * 2
    else:
        width_shell_separator = 63.7e-3
        # Mass of shell separator per unit length
        rho_shell_separator = rho_abs_sheet * width_shell_separator
        # Note: last 20mm of shell are unseparated
        L_unseparated = 20e-3
        mass_unseparated = L_unseparated * (rho_copper_tube * num_tubes + rho_acrylic_pipe)
        tube_length = L_unseparated + (mass_remainder - mass_unseparated) / (rho_shell_tubes + rho_shell_separator)
        tube_length = min(tube_length, tube_length_limit)
        baffle_length = tube_length - L_unseparated - L_shell_header
    baffle_spacing = baffle_length / (num_baffles - 1) # Note: breaks if num_baffles = 1. So don't do that.
    return tube_length, baffle_spacing

if __name__ == "__main__":
    num_tube_passes = 1
    num_shell_passes = 1
    num_tubes = 2
    num_baffles = 4
    tube_length, baffle_spacing = calculate_tube_length_baffle_spacing(num_shell_passes, \
                                                                       num_tube_passes, num_tubes, num_baffles)
    print(tube_length, baffle_spacing)
