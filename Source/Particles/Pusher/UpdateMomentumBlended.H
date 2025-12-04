/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BLENDED_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BLENDED_H_

#include <AMReX_REAL.H>
#include "Particles/Pusher/ComputeBlendedAlpha.H"

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBlended(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt,
    const amrex::ParticleReal gBx, const amrex::ParticleReal gBy, const amrex::ParticleReal gBz)
{
    using namespace amrex::literals;

    const amrex::ParticleReal econst = 0.5_prt*q*dt/m;

    // Compute blending coefficient alpha
    const amrex::ParticleReal alpha = ComputeBlendedAlpha(Bx, By, Bz, q, m, dt);

    // Compute temporary gamma factor
    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);

    // Blended pusher includes mirror force correction to momentum
    // compute mu using original momentum
    const amrex::ParticleReal Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);

    // Limit Bmin to avoid division by zero
    constexpr amrex::ParticleReal Bmin  = 1e-12_prt;     // Tesla
    constexpr amrex::ParticleReal B2min = Bmin * Bmin;
    const amrex::Real invB = 1.0_prt / amrex::max(Bmag, Bmin);

    const amrex::ParticleReal bx = Bx*invB;
    const amrex::ParticleReal by = By*invB;
    const amrex::ParticleReal bz = Bz*invB;

    //compute parallel and perpendicular velocities
    //const amrex::ParticleReal upar_x = ux*bx;
    //const amrex::ParticleReal upar_y = uy*by;
    //const amrex::ParticleReal upar_z = uz*bz;

    const amrex::ParticleReal uperp_x = ux - ux*bx;
    const amrex::ParticleReal uperp_y = uy - uy*by;
    const amrex::ParticleReal uperp_z = uz - uz*bz;

    const amrex::ParticleReal mu = m*inv_gamma*(uperp_x*uperp_x + uperp_y*uperp_y + uperp_z*uperp_z) *0.5_prt*invB;

    // First half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;
    // Magnetic rotation
    // - Compute temporary variables
    const amrex::ParticleReal tx = econst*inv_gamma*Bx;
    const amrex::ParticleReal ty = econst*inv_gamma*By;
    const amrex::ParticleReal tz = econst*inv_gamma*Bz;
    const amrex::ParticleReal tsqi = 2._prt/(1._prt + tx*tx + ty*ty + tz*tz);
    const amrex::ParticleReal sx = tx*tsqi;
    const amrex::ParticleReal sy = ty*tsqi;
    const amrex::ParticleReal sz = tz*tsqi;
    const amrex::ParticleReal ux_p = ux + uy*tz - uz*ty;
    const amrex::ParticleReal uy_p = uy + uz*tx - ux*tz;
    const amrex::ParticleReal uz_p = uz + ux*ty - uy*tx;
    // - Update momentum
    ux += uy_p*sz - uz_p*sy;
    uy += uz_p*sx - ux_p*sz;
    uz += ux_p*sy - uy_p*sx;
    // Second half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;

    // Mirror force correction to momentum
    const amrex::ParticleReal dux = -1._prt*(mu/m) * (bx*gBx) * dt
    const amrex::ParticleReal duy = -1._prt*(mu/m) * (by*gBy) * dt
    const amrex::ParticleReal duz = -1._prt*(mu/m) * (bz*gBz) * dt

    ux -= 0.5_prt*(1._prt - alpha)*dux;
    uy -= 0.5_prt*(1._prt - alpha)*duy;
    uz -= 0.5_prt*(1._prt - alpha)*duz;

}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BLENDED_H_
