#ifndef WARPX_HELIUM_HELIUM_FUSION_CROSS_SECTION_H
#define WARPX_HELIUM_HELIUM_FUSION_CROSS_SECTION_H

#include "Utils/WarpXConst.H"
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <cmath>


/**
 * \brief Computes the He3-He3 fusion cross section for high energies (E_cm >= 0.4 MeV)
 * using a linear interpolation of ENDF/B-VIII.0 evaluated data.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal HeliumHeliumFusionCrossSectionHighEnergy (const amrex::ParticleReal& E_MeV)
{
    using namespace amrex::literals;

    // Hardcoded ENDF/B-VIII.0 data points for E_cm >= 0.4 MeV
    // Data extracted and converted to E_cm (E_lab / 2). 
    // E_cm is in MeV, Cross section is in barns.
    constexpr int n_points = 16;
    constexpr amrex::GpuArray<amrex::ParticleReal, n_points> E_cm_table = {
        0.40_prt, 0.50_prt, 0.60_prt, 0.70_prt, 0.80_prt, 0.90_prt, 
        1.00_prt, 1.50_prt, 2.00_prt, 3.00_prt, 4.00_prt, 5.00_prt, 
        6.00_prt, 7.00_prt, 8.00_prt, 10.00_prt
    };
    constexpr amrex::GpuArray<amrex::ParticleReal, n_points> sigma_table = {
        0.0048292_prt, 0.0083371_prt, 0.012230_prt, 0.016327_prt, 0.020573_prt, 0.024993_prt,
        0.029658_prt,  0.048250_prt,  0.064040_prt, 0.083553_prt, 0.095072_prt, 0.104150_prt,
        0.110800_prt,  0.115010_prt,  0.116790_prt, 0.113030_prt
    };

    // If energy exceeds our highest tabulated point, return the last known value
    if (E_MeV >= E_cm_table[n_points - 1]) {
        return sigma_table[n_points - 1];
    }

    // Fast linear search to find the correct energy bin
    int idx = 0;
    for (int i = 0; i < n_points - 1; ++i) {
        if (E_MeV >= E_cm_table[i] && E_MeV < E_cm_table[i+1]) {
            idx = i;
            break;
        }
    }

    // Linear interpolation
    const amrex::ParticleReal E0 = E_cm_table[idx];
    const amrex::ParticleReal E1 = E_cm_table[idx+1];
    const amrex::ParticleReal sig0 = sigma_table[idx];
    const amrex::ParticleReal sig1 = sigma_table[idx+1];

    const amrex::ParticleReal slope = (sig1 - sig0) / (E1 - E0);
    return sig0 + slope * (E_MeV - E0);
}

/**
 * \brief Computes the He3-He3 fusion cross section for low energies 
 * using the Solar Fusion III recommendation.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal HeliumHeliumFusionCrossSectionSFIII (const amrex::ParticleReal& E_MeV)
{
    using namespace amrex::literals;

    // SF III S-factor in MeV b
    const amrex::ParticleReal S33 = 5.21_prt - 4.90_prt * E_MeV + 11.21_prt * E_MeV * E_MeV;

    // Sommerfeld parameter calculation (Z1=2, Z2=2)
    constexpr double m_He3 = 3.0160293_prt * PhysConst::m_u - 2.0_prt * PhysConst::m_e;
    
    // Explicitly define the fine structure constant
    constexpr amrex::ParticleReal alpha = 1.0_prt / 137.035999_prt;
    
    constexpr amrex::ParticleReal factor = 4.0_prt * MathConst::pi * alpha *
        std::sqrt(0.5_prt * static_cast<amrex::ParticleReal>(m_He3) * PhysConst::c * PhysConst::c / (1.e6_prt * PhysConst::q_e));
    
    const amrex::ParticleReal two_pi_eta = factor / std::sqrt(E_MeV);

    // Return cross section in barn
    return (S33 / E_MeV) * std::exp(-two_pi_eta);
}

/**
 * \brief Master function that routes the center-of-mass energy to the 
 * appropriate cross-section fit.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal HeliumHeliumFusionCrossSection (const amrex::ParticleReal& E_kin_star)
{
    using namespace amrex::literals;

    if (E_kin_star <= 0._prt) return 0._prt;

    // Convert energy from Joules to MeV
    constexpr amrex::ParticleReal joule_to_MeV = 1.e-6_prt / PhysConst::q_e;
    const amrex::ParticleReal E_MeV = E_kin_star * joule_to_MeV;

    // Transition energy shifted to 0.4 MeV to prevent SF-III polynomial blowup
    constexpr amrex::ParticleReal E_threshold = 0.4_prt; 

    // Route to the appropriate fit
    const amrex::ParticleReal cross_section_b = (E_MeV <= E_threshold) ?
                                                HeliumHeliumFusionCrossSectionSFIII(E_MeV) :
                                                HeliumHeliumFusionCrossSectionHighEnergy(E_MeV);

    // Convert barn to m^2
    constexpr amrex::ParticleReal barn_to_sqm = 1.e-28_prt;
    return cross_section_b * barn_to_sqm;
}
#endif // WARPX_HELIUM_HELIUM_FUSION_CROSS_SECTION_H