import numpy as np

# Material properties
rho_copper_tube = 0.2   # kg / m
rho_acrylic_pipe = 0.65 # kg / m
m_nozzle = 0.025        # kg
rho_abs_sheet = 2.39    # kg / m2
rho_3D_printed = 1150   # kg / m3
m_o_ring = 5.3e-3       # kg

mass_limit = 1
tube_length_limit_one_tube_pass = 0.35 - 40e-3 * 2 - 1.5e-3 * 4
tube_length_limit_even_tube_pass = 0.35 - 40e-3 - 20e-3 - 1.5e-3 * 4

# Calculate mass of unchanging parts
def _constant_mass(num_shell_passes, num_tube_passes, num_tubes, num_baffles):
    assert(num_shell_passes == 1 or num_shell_passes == 2)
    assert(num_tube_passes == 1 or num_tube_passes == 2 or num_tube_passes == 4)
    assert(num_tubes % num_tube_passes == 0)

    result = 0

    # End plates
    d_endplate = 69e-3  # m
    h_endplate = 4.5e-3 # m
    A_endplate = np.pi * d_endplate**2 / 4 
    result += 2 * rho_3D_printed * h_endplate * A_endplate

    # Header-shell separators
    h_separator = h_endplate
    od_tube = 8e-3
    A_tubes = num_tubes * np.pi * od_tube**2 / 4
    A_separator = A_endplate - A_tubes
    result += 2 * rho_3D_printed * h_separator * A_separator

    # O-rings
    result += 6 * m_o_ring

    # Nozzles
    result += 4 * m_nozzle

    # Baffles
    h_baffle = 1.5e-3
    # Conservative estimate, neglect the missing mass of the baffle cut
    A_baffle = A_separator
    result += num_baffles * rho_3D_printed * h_baffle * A_baffle

    # Front header
    l_small_header = 20e-3
    l_large_header = 40e-3
    result += l_large_header * rho_acrylic_pipe
    # Rear header
    if num_tube_passes == 1:
        result += l_large_header * rho_acrylic_pipe
    else:
        result += l_small_header * rho_acrylic_pipe
    # TODO account for header separators in case of multiple tube passes

    # Note: the fixed mass is the same, regardless of num_shell_passes
    return result

def calculate_tube_length_baffle_spacing(num_shell_passes, num_tube_passes, num_tubes, num_baffles):
    """Find the maximum length of tubes and baffle spacing for a given geometry configuration"""
    mass_remainder = mass_limit - _constant_mass(num_shell_passes, num_tube_passes, num_tubes, num_baffles)
    rho_shell_tubes = rho_copper_tube * num_tubes + rho_acrylic_pipe
    L_shell_header = 40e-3

    if num_tube_passes == 1:
        tube_length_limit = tube_length_limit_one_tube_pass
    else:
        tube_length_limit = tube_length_limit_even_tube_pass

    if num_shell_passes == 1:
        tube_length = mass_remainder / rho_shell_tubes
        tube_length = min(tube_length, tube_length_limit)
        baffle_length = tube_length - L_shell_header * 2
    else:
        h_shell_separator = 1.5e-3
        width_shell_separator = 63.7e-3
        # Mass of shell separator per unit length
        rho_shell_separator = rho_3D_printed * h_shell_separator * width_shell_separator
        # Note: last 20mm of shell are unseparated
        L_unseparated = 20e-3
        mass_unseparated = L_unseparated * (rho_copper_tube * num_tubes + rho_acrylic_pipe)
        tube_length = L_unseparated + (mass_remainder - mass_unseparated) / (rho_shell_tubes + rho_shell_separator)
        tube_length = min(tube_length, tube_length_limit)
        baffle_length = tube_length - L_unseparated - L_shell_header
    baffle_spacing = baffle_length / (num_baffles - 1) # Note: breaks if num_baffles = 1. So don't do that.
    return tube_length, baffle_spacing

# TODO test that this gives reasonable results

if __name__ == "__main__":
    num_tube_passes = 1
    num_shell_passes = 1
    num_tubes = 2
    num_baffles = 4
    tube_length, baffle_spacing = calculate_tube_length_baffle_spacing(num_shell_passes, \
                                                                       num_tube_passes, num_tubes, num_baffles)
    print(tube_length, baffle_spacing)