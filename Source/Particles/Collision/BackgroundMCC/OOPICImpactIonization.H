/*
 * Electron-impact-ionization final-state sampling adapted from XOOPIC's
 * relativistic MCC model (MCCPackage::ejectedEnergy,
 * MCCPackage::primarySecondaryAngles, and MCCPackage::newMomentum).
 */
#ifndef WARPX_PARTICLES_COLLISION_OOPIC_IMPACT_IONIZATION_H_
#define WARPX_PARTICLES_COLLISION_OOPIC_IMPACT_IONIZATION_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace OOPICImpactIonization
{
    using namespace amrex::literals;

    /** Sample the kinetic energy (eV) of the newly freed electron.
     *
     * This is the rejection sampler used by XOOPIC's relativistic MCC path,
     * based on its relativistic extension of the Rudd singly differential
     * ionization cross section.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal sampleSecondaryEnergy (
        amrex::ParticleReal const impact_energy_ev,
        amrex::ParticleReal const ionization_energy_ev,
        amrex::ParticleReal const electron_rest_energy_ev,
        amrex::RandomEngine const& engine) noexcept
    {
        using amrex::ParticleReal;

        const ParticleReal T = impact_energy_ev;
        const ParticleReal I = ionization_energy_ev;
        const ParticleReal mc2 = electron_rest_energy_ev;
        if (T <= I) { return 0.0_prt; }

        const ParticleReal t_plus_mc = T + mc2;
        const ParticleReal two_t_plus_mc = 2.0_prt*T + mc2;
        const ParticleReal t_plus_i = T + I;
        const ParticleReal t_minus_i = T - I;
        const ParticleReal inv_t_plus_mc_sq = 1.0_prt/(t_plus_mc*t_plus_mc);
        const ParticleReal i_over_t = I/T;

        const ParticleReal f1 = 14.0_prt/3.0_prt
            + 0.25_prt*t_plus_i*t_plus_i*inv_t_plus_mc_sq
            - mc2*two_t_plus_mc*i_over_t*inv_t_plus_mc_sq;
        const ParticleReal f2 = 5.0_prt/3.0_prt - i_over_t
            - 2.0_prt*i_over_t*i_over_t/3.0_prt
            + 0.5_prt*I*t_minus_i*inv_t_plus_mc_sq
            + mc2*two_t_plus_mc*I*inv_t_plus_mc_sq*std::log(i_over_t)/t_plus_i;
        const ParticleReal a_greater = f1*t_minus_i/(f2*t_plus_i);

        // Random() can return zero. Keeping the proposal strictly inside the
        // open interval avoids W=0 singularities in the angular sampler.
        constexpr ParticleReal eps = 1.0e-12_prt;
        while (true) {
            const ParticleReal random_f = amrex::max(amrex::Random(engine), eps)*a_greater;
            const ParticleReal W = I*f2*random_f/(f1-f2*random_f);
            const ParticleReal w_plus_i = W + I;
            const ParticleReal inv_t_minus_w = 1.0_prt/(T-W);
            const ParticleReal w_plus_i_sq = w_plus_i*w_plus_i;
            const ParticleReal ratio = (
                1.0_prt + 4.0_prt*I/(3.0_prt*w_plus_i)
                + w_plus_i_sq*inv_t_minus_w*inv_t_minus_w
                + 4.0_prt*I*w_plus_i_sq*inv_t_minus_w*inv_t_minus_w*inv_t_minus_w/3.0_prt
                - mc2*two_t_plus_mc*w_plus_i*inv_t_minus_w*inv_t_plus_mc_sq
                + w_plus_i_sq*inv_t_plus_mc_sq) / f1;
            if (ratio >= amrex::Random(engine)) {
                return amrex::min(W, t_minus_i);
            }
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal samplePolarAngle (
        amrex::ParticleReal const outgoing_energy_ev,
        amrex::ParticleReal const impact_energy_ev,
        amrex::ParticleReal const ionization_energy_ev,
        amrex::ParticleReal const electron_rest_energy_ev,
        amrex::RandomEngine const& engine) noexcept
    {
        using amrex::ParticleReal;
        const ParticleReal W = amrex::max(outgoing_energy_ev, 1.0e-12_prt);
        const ParticleReal T = impact_energy_ev;
        const ParticleReal I = ionization_energy_ev;
        const ParticleReal mc2 = electron_rest_energy_ev;
        const ParticleReal w_plus_i = W + I;
        const ParticleReal g2 = std::sqrt(
            w_plus_i*(T+2.0_prt*mc2)/((w_plus_i+2.0_prt*mc2)*T));
        ParticleReal alpha = mc2/(T+mc2);
        alpha *= 0.6_prt*alpha;
        const ParticleReal radicand = amrex::max(
            I*(1.0_prt-w_plus_i/T)/W, 0.0_prt);
        const ParticleReal g3 = alpha*std::sqrt(radicand);
        const ParticleReal f = amrex::Random(engine);
        const ParticleReal argument = g2 + g3*std::tan(
            (1.0_prt-f)*std::atan2(1.0_prt-g2, g3)
            - f*std::atan2(1.0_prt+g2, g3));
        return std::acos(amrex::max(-1.0_prt, amrex::min(1.0_prt, argument)));
    }

    /** Sample OOPIC's correlated primary/secondary scattering angles. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void sampleAngles (
        amrex::ParticleReal const impact_energy_ev,
        amrex::ParticleReal const secondary_energy_ev,
        amrex::ParticleReal const ionization_energy_ev,
        amrex::ParticleReal const electron_rest_energy_ev,
        amrex::ParticleReal& theta_primary,
        amrex::ParticleReal& phi_primary,
        amrex::ParticleReal& theta_secondary,
        amrex::ParticleReal& phi_secondary,
        amrex::RandomEngine const& engine) noexcept
    {
        const amrex::ParticleReal primary_energy_ev =
            amrex::max(impact_energy_ev-secondary_energy_ev-ionization_energy_ev, 0.0_prt);
        theta_primary = samplePolarAngle(primary_energy_ev, impact_energy_ev,
            ionization_energy_ev, electron_rest_energy_ev, engine);
        theta_secondary = samplePolarAngle(secondary_energy_ev, impact_energy_ev,
            ionization_energy_ev, electron_rest_energy_ev, engine);
        phi_primary = 2.0_prt*MathConst::pi*amrex::Random(engine);

        const amrex::ParticleReal critical_energy = 10.0_prt*ionization_energy_ev;
        amrex::ParticleReal correlation = 0.0_prt;
        if (secondary_energy_ev > critical_energy && primary_energy_ev > critical_energy) {
            correlation = std::sqrt(
                (secondary_energy_ev-critical_energy)*(primary_energy_ev-critical_energy))
                / critical_energy;
        }
        const amrex::ParticleReal delta = 2.0_prt*MathConst::pi/(1.0_prt+correlation);
        phi_secondary = phi_primary + MathConst::pi
            + (amrex::Random(engine)-0.5_prt)*delta;
    }

    /** Set proper velocity u=gamma*v at the requested energy and angles.
     *
     * Angles are measured relative to the incoming proper-velocity vector.
     * A full orthonormal basis is used here, fixing XOOPIC's loss of the sign
     * of the incoming y component in MCCPackage::newMomentum.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void setMomentum (
        amrex::ParticleReal const ux_in,
        amrex::ParticleReal const uy_in,
        amrex::ParticleReal const uz_in,
        amrex::ParticleReal const energy_ev,
        amrex::ParticleReal const electron_rest_energy_ev,
        amrex::ParticleReal const theta,
        amrex::ParticleReal const phi,
        amrex::ParticleReal& ux_out,
        amrex::ParticleReal& uy_out,
        amrex::ParticleReal& uz_out) noexcept
    {
        using amrex::ParticleReal;
        const ParticleReal u_in = std::sqrt(ux_in*ux_in + uy_in*uy_in + uz_in*uz_in);
        const ParticleReal nx = ux_in/u_in;
        const ParticleReal ny = uy_in/u_in;
        const ParticleReal nz = uz_in/u_in;

        ParticleReal e1x, e1y, e1z;
        if (std::abs(nz) < 0.9_prt) {
            const ParticleReal inv = 1.0_prt/std::sqrt(nx*nx+ny*ny);
            e1x = -ny*inv; e1y = nx*inv; e1z = 0.0_prt;
        } else {
            const ParticleReal inv = 1.0_prt/std::sqrt(ny*ny+nz*nz);
            e1x = 0.0_prt; e1y = -nz*inv; e1z = ny*inv;
        }
        const ParticleReal e2x = ny*e1z-nz*e1y;
        const ParticleReal e2y = nz*e1x-nx*e1z;
        const ParticleReal e2z = nx*e1y-ny*e1x;

        const ParticleReal sin_theta = std::sin(theta);
        const ParticleReal cos_theta = std::cos(theta);
        const ParticleReal cos_phi = std::cos(phi);
        const ParticleReal sin_phi = std::sin(phi);
        const ParticleReal gamma_minus_one = energy_ev/electron_rest_energy_ev;
        const ParticleReal u_magnitude = PhysConst::c*std::sqrt(
            gamma_minus_one*(gamma_minus_one+2.0_prt));
        ux_out = u_magnitude*(cos_theta*nx + sin_theta*(cos_phi*e1x+sin_phi*e2x));
        uy_out = u_magnitude*(cos_theta*ny + sin_theta*(cos_phi*e1y+sin_phi*e2y));
        uz_out = u_magnitude*(cos_theta*nz + sin_theta*(cos_phi*e1z+sin_phi*e2z));
    }
}

#endif
