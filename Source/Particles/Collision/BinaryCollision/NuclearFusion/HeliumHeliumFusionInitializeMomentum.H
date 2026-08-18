/* Copyright 2024
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_HELIUM_HELIUM_FUSION_INITIALIZE_MOMENTUM_H
#define WARPX_HELIUM_HELIUM_FUSION_INITIALIZE_MOMENTUM_H

#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_DenseBins.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace {
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<SoaData_type>;
    using index_type = typename ParticleBins::index_type;

    /**
     * \brief Initializes the momenta of the alpha particle and two protons produced from
     * He3-He3 fusion. The process is modeled as a direct three-body breakup.
     * Energy is distributed among the three products according to non-relativistic 
     * three-body phase space, and particles are emitted isotropically in their 
     * respective center-of-mass frames.
     *
     * @param[in] soa_1 struct of array data of the first He3 colliding macroparticle
     * @param[in] soa_2 struct of array data of the second He3 colliding macroparticle
     * @param[out] soa_alpha struct of array data of the product alpha species
     * @param[out] soa_proton struct of array data of the product proton species
     * @param[in] idx_1 index of first colliding macroparticle
     * @param[in] idx_2 index of second colliding macroparticle
     * @param[in] idx_alpha_start index of first produced alpha macroparticle
     * @param[in] idx_proton_start index of first produced proton macroparticle
     * @param[in] m1 mass of first colliding species (He3)
     * @param[in] m2 mass of second colliding species (He3)
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void HeliumHeliumFusionInitializeMomentum (
                            const SoaData_type& soa_1, const SoaData_type& soa_2,
                            SoaData_type& soa_alpha, SoaData_type& soa_proton,
                            const index_type& idx_1, const index_type& idx_2,
                            const index_type& idx_alpha_start,
                            const index_type& idx_proton_start,
                            const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                            const amrex::RandomEngine& engine)
    {
        using namespace amrex::literals;
        
        // Enforce double precision for all kinematic calculations
        constexpr double c2_dbl = PhysConst::c * PhysConst::c;
        constexpr double inv_c_dbl = 1.0 / PhysConst::c;

        constexpr double m_alpha_dbl = PhysConst::m_u * 4.00260325413 - 2.0 * PhysConst::m_e;
        constexpr double m_proton_dbl = PhysConst::m_p;
        
        const double m1_dbl = static_cast<double>(m1);
        const double m2_dbl = static_cast<double>(m2);

        // Extract momenta of the reacting He3 macroparticles (u = p / m)
        const double u1x_dbl = static_cast<double>(soa_1.m_rdata[PIdx::ux][idx_1]);
        const double u1y_dbl = static_cast<double>(soa_1.m_rdata[PIdx::uy][idx_1]);
        const double u1z_dbl = static_cast<double>(soa_1.m_rdata[PIdx::uz][idx_1]);
        const double g1_dbl = std::sqrt(1.0 + (u1x_dbl*u1x_dbl + u1y_dbl*u1y_dbl + u1z_dbl*u1z_dbl)*inv_c_dbl*inv_c_dbl);

        const double u2x_dbl = static_cast<double>(soa_2.m_rdata[PIdx::ux][idx_2]);
        const double u2y_dbl = static_cast<double>(soa_2.m_rdata[PIdx::uy][idx_2]);
        const double u2z_dbl = static_cast<double>(soa_2.m_rdata[PIdx::uz][idx_2]);
        const double g2_dbl = std::sqrt(1.0 + (u2x_dbl*u2x_dbl + u2y_dbl*u2y_dbl + u2z_dbl*u2z_dbl)*inv_c_dbl*inv_c_dbl);

        // Calculate total energy and momentum in the lab frame
        const double E_lab_tot_dbl = (g1_dbl * m1_dbl + g2_dbl * m2_dbl) * c2_dbl;
        const double px_lab_tot_dbl = m1_dbl * u1x_dbl + m2_dbl * u2x_dbl;
        const double py_lab_tot_dbl = m1_dbl * u1y_dbl + m2_dbl * u2y_dbl;
        const double pz_lab_tot_dbl = m1_dbl * u1z_dbl + m2_dbl * u2z_dbl;
        const double p_lab_tot_sq_dbl = px_lab_tot_dbl*px_lab_tot_dbl + py_lab_tot_dbl*py_lab_tot_dbl + pz_lab_tot_dbl*pz_lab_tot_dbl;

        // Invariant mass of the system
        const double M_inv_dbl = std::sqrt(std::max(0.0, E_lab_tot_dbl*E_lab_tot_dbl*inv_c_dbl*inv_c_dbl - p_lab_tot_sq_dbl)) * inv_c_dbl;
        const double Q_tot_dbl = (M_inv_dbl - (m_alpha_dbl + 2.0*m_proton_dbl)) * c2_dbl;

        // -------------------------------------------------------------------------
        // 1. Phase Space Sampling for Energy Distribution
        amrex::ParticleReal x_rnd, y_rnd;
        do {
            x_rnd = amrex::Random(engine);
            y_rnd = 0.5_prt * amrex::Random(engine); 
        } while (y_rnd > std::sqrt(x_rnd * (1.0_prt - x_rnd)));
        
        const double x = static_cast<double>(x_rnd);
        const double E_pp_kin = x * Q_tot_dbl; // CM kinetic energy of the 2-proton system

        // Exact relativistic mass of the 2-proton system
        const double mass_pp = 2.0*m_proton_dbl + E_pp_kin*inv_c_dbl*inv_c_dbl;

        // -------------------------------------------------------------------------
        // 2. Exact Relativistic 2-Body Decay in Total CM Frame
        const double E_alpha_cm = (M_inv_dbl*M_inv_dbl*c2_dbl*c2_dbl + m_alpha_dbl*m_alpha_dbl*c2_dbl*c2_dbl - mass_pp*mass_pp*c2_dbl*c2_dbl) / (2.0 * M_inv_dbl * c2_dbl);
        const double p_alpha_cm_sq = std::max(0.0, E_alpha_cm*E_alpha_cm*inv_c_dbl*inv_c_dbl - m_alpha_dbl*m_alpha_dbl*c2_dbl);
        const double p_alpha_cm_mag = std::sqrt(p_alpha_cm_sq);
        
        amrex::ParticleReal px_alpha_cm_rt, py_alpha_cm_rt, pz_alpha_cm_rt;
        ParticleUtils::RandomizeVelocity(px_alpha_cm_rt, py_alpha_cm_rt, pz_alpha_cm_rt, static_cast<amrex::ParticleReal>(p_alpha_cm_mag), engine);
        const double px_alpha_cm = static_cast<double>(px_alpha_cm_rt);
        const double py_alpha_cm = static_cast<double>(py_alpha_cm_rt);
        const double pz_alpha_cm = static_cast<double>(pz_alpha_cm_rt);

        const double px_pp_cm = -px_alpha_cm;
        const double py_pp_cm = -py_alpha_cm;
        const double pz_pp_cm = -pz_alpha_cm;

        // -------------------------------------------------------------------------
        // 3. Exact Relativistic Proton Kinematics in the 2-Proton CM Frame
        const double E_p1_star = mass_pp * c2_dbl / 2.0;
        const double p_proton_star_sq = std::max(0.0, E_p1_star*E_p1_star*inv_c_dbl*inv_c_dbl - m_proton_dbl*m_proton_dbl*c2_dbl);
        const double p_proton_star_mag = std::sqrt(p_proton_star_sq);
        
        amrex::ParticleReal px_p1_star_rt, py_p1_star_rt, pz_p1_star_rt;
        ParticleUtils::RandomizeVelocity(px_p1_star_rt, py_p1_star_rt, pz_p1_star_rt, static_cast<amrex::ParticleReal>(p_proton_star_mag), engine);
        const double px_p1_star = static_cast<double>(px_p1_star_rt);
        const double py_p1_star = static_cast<double>(py_p1_star_rt);
        const double pz_p1_star = static_cast<double>(pz_p1_star_rt);

        const double px_p2_star = -px_p1_star;
        const double py_p2_star = -py_p1_star;
        const double pz_p2_star = -pz_p1_star;

        // -------------------------------------------------------------------------
        // 4. Boost Protons from PP CM Frame to the Total CM Frame
        const double E_pp_cm = std::sqrt(mass_pp*mass_pp*c2_dbl*c2_dbl + (px_pp_cm*px_pp_cm + py_pp_cm*py_pp_cm + pz_pp_cm*pz_pp_cm)*c2_dbl);
        const double g_pp = E_pp_cm / (mass_pp * c2_dbl);
        const double vx_pp = px_pp_cm * c2_dbl / E_pp_cm;
        const double vy_pp = py_pp_cm * c2_dbl / E_pp_cm;
        const double vz_pp = pz_pp_cm * c2_dbl / E_pp_cm;
        const double v_pp_sq = vx_pp*vx_pp + vy_pp*vy_pp + vz_pp*vz_pp;

        double px_p1_cm, py_p1_cm, pz_p1_cm;
        double px_p2_cm, py_p2_cm, pz_p2_cm;

        if (v_pp_sq > 1.0e-20) {
            const double v_dot_p1 = vx_pp*px_p1_star + vy_pp*py_p1_star + vz_pp*pz_p1_star;
            const double factor1 = (g_pp - 1.0) / v_pp_sq * v_dot_p1 + g_pp * E_p1_star * inv_c_dbl * inv_c_dbl;
            px_p1_cm = px_p1_star + vx_pp * factor1;
            py_p1_cm = py_p1_star + vy_pp * factor1;
            pz_p1_cm = pz_p1_star + vz_pp * factor1;

            const double v_dot_p2 = vx_pp*px_p2_star + vy_pp*py_p2_star + vz_pp*pz_p2_star;
            const double factor2 = (g_pp - 1.0) / v_pp_sq * v_dot_p2 + g_pp * E_p1_star * inv_c_dbl * inv_c_dbl;
            px_p2_cm = px_p2_star + vx_pp * factor2;
            py_p2_cm = py_p2_star + vy_pp * factor2;
            pz_p2_cm = pz_p2_star + vz_pp * factor2;
        } else {
            px_p1_cm = px_p1_star; py_p1_cm = py_p1_star; pz_p1_cm = pz_p1_star;
            px_p2_cm = px_p2_star; py_p2_cm = py_p2_star; pz_p2_cm = pz_p2_star;
        }

        // -------------------------------------------------------------------------
        // 5. Boost all Products to the Lab Frame
        const double bx = px_lab_tot_dbl * c2_dbl / E_lab_tot_dbl;
        const double by = py_lab_tot_dbl * c2_dbl / E_lab_tot_dbl;
        const double bz = pz_lab_tot_dbl * c2_dbl / E_lab_tot_dbl;
        const double b_sq = bx*bx + by*by + bz*bz;
        const double g_lab = E_lab_tot_dbl / (M_inv_dbl * c2_dbl);

        double px_alpha, py_alpha, pz_alpha;
        if (b_sq > 1.0e-20) {
            const double b_dot_p = bx*px_alpha_cm + by*py_alpha_cm + bz*pz_alpha_cm;
            const double factor = (g_lab - 1.0) / b_sq * b_dot_p + g_lab * E_alpha_cm * inv_c_dbl * inv_c_dbl;
            px_alpha = px_alpha_cm + bx * factor;
            py_alpha = py_alpha_cm + by * factor;
            pz_alpha = pz_alpha_cm + bz * factor;
        } else {
            px_alpha = px_alpha_cm; py_alpha = py_alpha_cm; pz_alpha = pz_alpha_cm;
        }

        double px_p1, py_p1, pz_p1;
        if (b_sq > 1.0e-20) {
            const double b_dot_p = bx*px_p1_cm + by*py_p1_cm + bz*pz_p1_cm;
            const double E_p1_cm = std::sqrt(m_proton_dbl*m_proton_dbl*c2_dbl*c2_dbl + (px_p1_cm*px_p1_cm + py_p1_cm*py_p1_cm + pz_p1_cm*pz_p1_cm)*c2_dbl);
            const double factor = (g_lab - 1.0) / b_sq * b_dot_p + g_lab * E_p1_cm * inv_c_dbl * inv_c_dbl;
            px_p1 = px_p1_cm + bx * factor;
            py_p1 = py_p1_cm + by * factor;
            pz_p1 = pz_p1_cm + bz * factor;
        } else {
            px_p1 = px_p1_cm; py_p1 = py_p1_cm; pz_p1 = pz_p1_cm;
        }

        double px_p2, py_p2, pz_p2;
        if (b_sq > 1.0e-20) {
            const double b_dot_p = bx*px_p2_cm + by*py_p2_cm + bz*pz_p2_cm;
            const double E_p2_cm = std::sqrt(m_proton_dbl*m_proton_dbl*c2_dbl*c2_dbl + (px_p2_cm*px_p2_cm + py_p2_cm*py_p2_cm + pz_p2_cm*pz_p2_cm)*c2_dbl);
            const double factor = (g_lab - 1.0) / b_sq * b_dot_p + g_lab * E_p2_cm * inv_c_dbl * inv_c_dbl;
            px_p2 = px_p2_cm + bx * factor;
            py_p2 = py_p2_cm + by * factor;
            pz_p2 = pz_p2_cm + bz * factor;
        } else {
            px_p2 = px_p2_cm; py_p2 = py_p2_cm; pz_p2 = pz_p2_cm;
        }

        // -------------------------------------------------------------------------
        // 6. Populate Output Containers (u = p / m)
        const amrex::ParticleReal ux_alpha = static_cast<amrex::ParticleReal>(px_alpha / m_alpha_dbl);
        const amrex::ParticleReal uy_alpha = static_cast<amrex::ParticleReal>(py_alpha / m_alpha_dbl);
        const amrex::ParticleReal uz_alpha = static_cast<amrex::ParticleReal>(pz_alpha / m_alpha_dbl);

        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start] = ux_alpha;
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start] = uy_alpha;
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start] = uz_alpha;
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 1] = ux_alpha;
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 1] = uy_alpha;
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 1] = uz_alpha;

        const amrex::ParticleReal ux_p1 = static_cast<amrex::ParticleReal>(px_p1 / m_proton_dbl);
        const amrex::ParticleReal uy_p1 = static_cast<amrex::ParticleReal>(py_p1 / m_proton_dbl);
        const amrex::ParticleReal uz_p1 = static_cast<amrex::ParticleReal>(pz_p1 / m_proton_dbl);

        const amrex::ParticleReal ux_p2 = static_cast<amrex::ParticleReal>(px_p2 / m_proton_dbl);
        const amrex::ParticleReal uy_p2 = static_cast<amrex::ParticleReal>(py_p2 / m_proton_dbl);
        const amrex::ParticleReal uz_p2 = static_cast<amrex::ParticleReal>(pz_p2 / m_proton_dbl);

        soa_proton.m_rdata[PIdx::ux][idx_proton_start]     = ux_p1;
        soa_proton.m_rdata[PIdx::uy][idx_proton_start]     = uy_p1;
        soa_proton.m_rdata[PIdx::uz][idx_proton_start]     = uz_p1;
        soa_proton.m_rdata[PIdx::ux][idx_proton_start + 1] = ux_p2;
        soa_proton.m_rdata[PIdx::uy][idx_proton_start + 1] = uy_p2;
        soa_proton.m_rdata[PIdx::uz][idx_proton_start + 1] = uz_p2;

        soa_proton.m_rdata[PIdx::ux][idx_proton_start + 2] = ux_p1;
        soa_proton.m_rdata[PIdx::uy][idx_proton_start + 2] = uy_p1;
        soa_proton.m_rdata[PIdx::uz][idx_proton_start + 2] = uz_p1;
        soa_proton.m_rdata[PIdx::ux][idx_proton_start + 3] = ux_p2;
        soa_proton.m_rdata[PIdx::uy][idx_proton_start + 3] = uy_p2;
        soa_proton.m_rdata[PIdx::uz][idx_proton_start + 3] = uz_p2;
    }
}
#endif // WARPX_HELIUM_HELIUM_FUSION_INITIALIZE_MOMENTUM_H