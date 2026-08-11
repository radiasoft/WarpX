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

        constexpr amrex::ParticleReal mev_to_joule = PhysConst::q_e * 1.e6_prt;
        constexpr amrex::ParticleReal c2 = PhysConst::c * PhysConst::c;
        constexpr amrex::ParticleReal inv_c = 1.0_prt / PhysConst::c;

        // Q-value of the He3 + He3 -> alpha + 2p reaction
        constexpr amrex::ParticleReal E_fusion = 12.8589_prt * mev_to_joule;

        constexpr double m_alpha = PhysConst::m_u * 4.00260325413_prt - 2.0_prt * PhysConst::m_e;
        constexpr double m_proton = PhysConst::m_p;
        constexpr double m_He3 = 3.0160293_prt * PhysConst::m_u - 2.0_prt * PhysConst::m_e;

        // Extract momenta of the reacting He3 macroparticles (u = gamma * v = p / m)
        const amrex::ParticleReal u1x = soa_1.m_rdata[PIdx::ux][idx_1];
        const amrex::ParticleReal u1y = soa_1.m_rdata[PIdx::uy][idx_1];
        const amrex::ParticleReal u1z = soa_1.m_rdata[PIdx::uz][idx_1];
        const amrex::ParticleReal g1 = std::sqrt(1.0_prt + (u1x*u1x + u1y*u1y + u1z*u1z)*inv_c*inv_c);

        const amrex::ParticleReal u2x = soa_2.m_rdata[PIdx::ux][idx_2];
        const amrex::ParticleReal u2y = soa_2.m_rdata[PIdx::uy][idx_2];
        const amrex::ParticleReal u2z = soa_2.m_rdata[PIdx::uz][idx_2];
        const amrex::ParticleReal g2 = std::sqrt(1.0_prt + (u2x*u2x + u2y*u2y + u2z*u2z)*inv_c*inv_c);

        // Calculate total energy and momentum in the lab frame
        const amrex::ParticleReal E_lab_tot = (g1 * m1 + g2 * m2) * c2;
        const amrex::ParticleReal px_lab_tot = m1 * u1x + m2 * u2x;
        const amrex::ParticleReal py_lab_tot = m1 * u1y + m2 * u2y;
        const amrex::ParticleReal pz_lab_tot = m1 * u1z + m2 * u2z;

        const amrex::ParticleReal p_lab_tot_sq = px_lab_tot*px_lab_tot + py_lab_tot*py_lab_tot + pz_lab_tot*pz_lab_tot;

        // Invariant mass of the system
        const amrex::ParticleReal M_inv = std::sqrt(E_lab_tot*E_lab_tot*inv_c*inv_c - p_lab_tot_sq) * inv_c;

        // Total kinetic energy available for the products in the CM frame
        const amrex::ParticleReal Q_tot = (M_inv - static_cast<amrex::ParticleReal>(m_alpha + 2.0*m_proton)) * c2 + E_fusion;

        // -------------------------------------------------------------------------
        // 1. Phase Space Sampling for Energy Distribution
        // Using non-relativistic 3-body phase space: P(E_pp) ~ sqrt(E_pp * (Q_tot - E_pp))
        amrex::ParticleReal x, y;
        do {
            x = amrex::Random(engine);
            y = 0.5_prt * amrex::Random(engine); // Maximum of sqrt(x*(1-x)) is 0.5
        } while (y > std::sqrt(x * (1.0_prt - x)));

        const amrex::ParticleReal E_pp = x * Q_tot; // CM kinetic energy of the 2-proton system
        const amrex::ParticleReal E_alpha = (1.0_prt - x) * Q_tot; // CM kinetic energy of the alpha

        // -------------------------------------------------------------------------
        // 2. Alpha Particle Kinematics in the Total CM Frame
        const amrex::ParticleReal p_alpha_cm = std::sqrt(2.0_prt * static_cast<amrex::ParticleReal>(m_alpha) * E_alpha);
        
        amrex::ParticleReal px_alpha_cm, py_alpha_cm, pz_alpha_cm;
        ParticleUtils::RandomizeVelocity(px_alpha_cm, py_alpha_cm, pz_alpha_cm, p_alpha_cm, engine);

        // -------------------------------------------------------------------------
        // 3. Proton Kinematics in the 2-Proton CM Frame
        const amrex::ParticleReal p_proton_star = std::sqrt(static_cast<amrex::ParticleReal>(m_proton) * E_pp);
        
        amrex::ParticleReal px_p1_star, py_p1_star, pz_p1_star;
        ParticleUtils::RandomizeVelocity(px_p1_star, py_p1_star, pz_p1_star, p_proton_star, engine);

        // Proton 2 is back-to-back with Proton 1 in the 2-proton CM frame
        const amrex::ParticleReal px_p2_star = -px_p1_star;
        const amrex::ParticleReal py_p2_star = -py_p1_star;
        const amrex::ParticleReal pz_p2_star = -pz_p1_star;

        // -------------------------------------------------------------------------
        // 4. Boost Protons to the Total CM Frame
        // The 2-proton system momentum in total CM is exactly opposite to the alpha
        const amrex::ParticleReal px_pp_cm = -px_alpha_cm;
        const amrex::ParticleReal py_pp_cm = -py_alpha_cm;
        const amrex::ParticleReal pz_pp_cm = -pz_alpha_cm;
        
        const amrex::ParticleReal mass_pp_sys = static_cast<amrex::ParticleReal>(2.0*m_proton) + E_pp*inv_c*inv_c;
        const amrex::ParticleReal g_pp = std::sqrt(1.0_prt + (px_pp_cm*px_pp_cm + py_pp_cm*py_pp_cm + pz_pp_cm*pz_pp_cm)/(mass_pp_sys*mass_pp_sys*c2));
        
        const amrex::ParticleReal vx_pp = px_pp_cm / (g_pp * mass_pp_sys);
        const amrex::ParticleReal vy_pp = py_pp_cm / (g_pp * mass_pp_sys);
        const amrex::ParticleReal vz_pp = pz_pp_cm / (g_pp * mass_pp_sys);
        const amrex::ParticleReal v_pp_sq = vx_pp*vx_pp + vy_pp*vy_pp + vz_pp*vz_pp;

        amrex::ParticleReal px_p1_cm, py_p1_cm, pz_p1_cm;
        amrex::ParticleReal px_p2_cm, py_p2_cm, pz_p2_cm;

        if (v_pp_sq > 0.0_prt) {
            const amrex::ParticleReal v_dot_p1 = vx_pp*px_p1_star + vy_pp*py_p1_star + vz_pp*pz_p1_star;
            const amrex::ParticleReal factor1 = (g_pp - 1.0_prt) / v_pp_sq * v_dot_p1 + static_cast<amrex::ParticleReal>(m_proton) * g_pp;
            px_p1_cm = px_p1_star + vx_pp * factor1;
            py_p1_cm = py_p1_star + vy_pp * factor1;
            pz_p1_cm = pz_p1_star + vz_pp * factor1;

            const amrex::ParticleReal v_dot_p2 = vx_pp*px_p2_star + vy_pp*py_p2_star + vz_pp*pz_p2_star;
            const amrex::ParticleReal factor2 = (g_pp - 1.0_prt) / v_pp_sq * v_dot_p2 + static_cast<amrex::ParticleReal>(m_proton) * g_pp;
            px_p2_cm = px_p2_star + vx_pp * factor2;
            py_p2_cm = py_p2_star + vy_pp * factor2;
            pz_p2_cm = pz_p2_star + vz_pp * factor2;
        } else {
            px_p1_cm = px_p1_star; py_p1_cm = py_p1_star; pz_p1_cm = pz_p1_star;
            px_p2_cm = px_p2_star; py_p2_cm = py_p2_star; pz_p2_cm = pz_p2_star;
        }

        // -------------------------------------------------------------------------
        // 5. Boost all Products to the Lab Frame
        const amrex::ParticleReal bx = px_lab_tot / E_lab_tot * c2;
        const amrex::ParticleReal by = py_lab_tot / E_lab_tot * c2;
        const amrex::ParticleReal bz = pz_lab_tot / E_lab_tot * c2;
        const amrex::ParticleReal b_sq = bx*bx + by*by + bz*bz;
        const amrex::ParticleReal g_lab = 1.0_prt / std::sqrt(1.0_prt - b_sq*inv_c*inv_c);

        auto apply_boost = [=] (amrex::ParticleReal px_cm, amrex::ParticleReal py_cm, amrex::ParticleReal pz_cm, double mass, 
                                amrex::ParticleReal& px_lab, amrex::ParticleReal& py_lab, amrex::ParticleReal& pz_lab) {
            if (b_sq > 0.0_prt) {
                const amrex::ParticleReal b_dot_p = bx*px_cm + by*py_cm + bz*pz_cm;
                const amrex::ParticleReal E_cm = std::sqrt(static_cast<amrex::ParticleReal>(mass*mass)*c2*c2 + (px_cm*px_cm + py_cm*py_cm + pz_cm*pz_cm)*c2);
                const amrex::ParticleReal factor = (g_lab - 1.0_prt) / b_sq * b_dot_p + g_lab * E_cm * inv_c * inv_c;
                px_lab = px_cm + bx * factor;
                py_lab = py_cm + by * factor;
                pz_lab = pz_cm + bz * factor;
            } else {
                px_lab = px_cm; py_lab = py_cm; pz_lab = pz_cm;
            }
        };

        amrex::ParticleReal px_alpha, py_alpha, pz_alpha;
        amrex::ParticleReal px_p1, py_p1, pz_p1;
        amrex::ParticleReal px_p2, py_p2, pz_p2;

        apply_boost(px_alpha_cm, py_alpha_cm, pz_alpha_cm, m_alpha, px_alpha, py_alpha, pz_alpha);
        apply_boost(px_p1_cm, py_p1_cm, pz_p1_cm, m_proton, px_p1, py_p1, pz_p1);
        apply_boost(px_p2_cm, py_p2_cm, pz_p2_cm, m_proton, px_p2, py_p2, pz_p2);

        // -------------------------------------------------------------------------
        // 6. Populate Output Containers (u = p / m)
        // Note: For charge/weight conservation, we create 2 alphas and 4 protons 
        // per reacting pair (1 alpha + 2 protons at position 1, 1 alpha + 2 protons at position 2).

        const amrex::ParticleReal ux_alpha = px_alpha / static_cast<amrex::ParticleReal>(m_alpha);
        const amrex::ParticleReal uy_alpha = py_alpha / static_cast<amrex::ParticleReal>(m_alpha);
        const amrex::ParticleReal uz_alpha = pz_alpha / static_cast<amrex::ParticleReal>(m_alpha);

        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start]     = ux_alpha;
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start]     = uy_alpha;
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start]     = uz_alpha;
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 1] = ux_alpha;
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 1] = uy_alpha;
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 1] = uz_alpha;

        const amrex::ParticleReal ux_p1 = px_p1 / static_cast<amrex::ParticleReal>(m_proton);
        const amrex::ParticleReal uy_p1 = py_p1 / static_cast<amrex::ParticleReal>(m_proton);
        const amrex::ParticleReal uz_p1 = pz_p1 / static_cast<amrex::ParticleReal>(m_proton);

        const amrex::ParticleReal ux_p2 = px_p2 / static_cast<amrex::ParticleReal>(m_proton);
        const amrex::ParticleReal uy_p2 = py_p2 / static_cast<amrex::ParticleReal>(m_proton);
        const amrex::ParticleReal uz_p2 = pz_p2 / static_cast<amrex::ParticleReal>(m_proton);

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