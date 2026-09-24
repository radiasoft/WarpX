#ifndef WARPX_HELIUM_HELIUM_FUSION_CROSS_SECTION_H
#define WARPX_HELIUM_HELIUM_FUSION_CROSS_SECTION_H

#include "Utils/WarpXConst.H"
#include <AMReX_Array.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <cmath>


/**
 * \brief Computes the He3-He3 fusion cross section for high energies (E_cm >= 0.2 MeV)
 * using linear interpolation of the supplied ENDF/B-VII.1 data.
 * https://www.nndc.bnl.gov/sigma/index.jsp
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::ParticleReal HeliumHeliumFusionCrossSectionHighEnergy (const amrex::ParticleReal& E_MeV)
{
    using namespace amrex::literals;

    // All supplied ENDF export points for E_cm >= 0.2 MeV.
    // Reference: Examples/Tests/nuclear_fusion/helium_helium_cross_section_endf.csv
    // Keep the full grid to avoid additional error from subsampling the export.
    // Data extracted and converted to E_cm (E_lab / 2).
    // E_cm is in MeV, Cross section is in barns.
    constexpr int n_points = 45;
    constexpr amrex::GpuArray<amrex::ParticleReal, n_points> E_cm_table = {
        0.200_prt, 0.225_prt, 0.250_prt, 0.275_prt, 0.300_prt,
        0.350_prt, 0.400_prt, 0.450_prt, 0.500_prt, 0.550_prt,
        0.600_prt, 0.650_prt, 0.700_prt, 0.750_prt, 0.800_prt,
        0.850_prt, 0.900_prt, 0.950_prt, 1.000_prt, 1.250_prt,
        1.500_prt, 1.750_prt, 2.000_prt, 2.250_prt, 2.500_prt,
        2.750_prt, 3.000_prt, 3.250_prt, 3.500_prt, 3.750_prt,
        4.000_prt, 4.250_prt, 4.500_prt, 4.750_prt, 5.000_prt,
        5.500_prt, 6.000_prt, 6.500_prt, 7.000_prt, 7.500_prt,
        8.000_prt, 8.500_prt, 9.000_prt, 9.500_prt, 10.000_prt
    };
    constexpr amrex::GpuArray<amrex::ParticleReal, n_points> sigma_table = {
        4.3937000e-04_prt, 7.1908000e-04_prt, 1.0825000e-03_prt, 1.5290000e-03_prt,
        2.0552000e-03_prt, 3.3227000e-03_prt, 4.8292000e-03_prt, 6.5172000e-03_prt,
        8.3371000e-03_prt, 1.0251000e-02_prt, 1.2230000e-02_prt, 1.4259000e-02_prt,
        1.6327000e-02_prt, 1.8431000e-02_prt, 2.0573000e-02_prt, 2.2758000e-02_prt,
        2.4993000e-02_prt, 2.7289000e-02_prt, 2.9658000e-02_prt, 3.8880000e-02_prt,
        4.8250000e-02_prt, 5.6430000e-02_prt, 6.4040000e-02_prt, 7.0470000e-02_prt,
        7.6020000e-02_prt, 8.0293000e-02_prt, 8.3553000e-02_prt, 8.6661000e-02_prt,
        8.9617000e-02_prt, 9.2421000e-02_prt, 9.5072000e-02_prt, 9.7571000e-02_prt,
        9.9918000e-02_prt, 1.0211000e-01_prt, 1.0415000e-01_prt, 1.0778000e-01_prt,
        1.1080000e-01_prt, 1.1321000e-01_prt, 1.1501000e-01_prt, 1.1621000e-01_prt,
        1.1679000e-01_prt, 1.1676000e-01_prt, 1.1613000e-01_prt, 1.1489000e-01_prt,
        1.1303000e-01_prt
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

    const amrex::ParticleReal factor = 4.0_prt * MathConst::pi * alpha *
        std::sqrt(static_cast<amrex::ParticleReal>(m_He3) * PhysConst::c * PhysConst::c / (1.e6_prt * PhysConst::q_e));

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

    if (E_kin_star <= 0._prt) { return 0._prt; }

    // Convert energy from Joules to MeV
    constexpr amrex::ParticleReal joule_to_MeV = 1.e-6_prt / PhysConst::q_e;
    const amrex::ParticleReal E_MeV = E_kin_star * joule_to_MeV;

    // At 0.2 MeV the fits differ by about 1.6%, versus 16.8% at 0.4 MeV.
    constexpr amrex::ParticleReal E_threshold = 0.2_prt;

    // Route to the appropriate fit
    const amrex::ParticleReal cross_section_b = (E_MeV <= E_threshold) ?
                                                HeliumHeliumFusionCrossSectionSFIII(E_MeV) :
                                                HeliumHeliumFusionCrossSectionHighEnergy(E_MeV);

    // Convert barn to m^2
    constexpr amrex::ParticleReal barn_to_sqm = 1.e-28_prt;
    return cross_section_b * barn_to_sqm;
}
#endif // WARPX_HELIUM_HELIUM_FUSION_CROSS_SECTION_H
