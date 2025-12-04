/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEPOSITIONBLENDED_H_
#define WARPX_PARTICLES_PUSHER_UPDATEPOSITIONBLENDED_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>



/** \brief Push the particle position over one time step,
 *         given the value of its momenta `ux`, `uy`, `uz`,
 *         magnetic field components, `Bx`, `By`, `Bz`,
 *         and magnetic field gradients `gBx`, `gBy`, `gBz`,
 *         using the Blended position update, including estimated drifts (gradB, ExB, curvature)
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePositionBlended ([[maybe_unused]] amrex::ParticleReal& x,
                     [[maybe_unused]] amrex::ParticleReal& y,
                     [[maybe_unused]] amrex::ParticleReal& z,
                     const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
                     const amrex::Real dt,
                     amrex::ParticleReal const m, const amrex::ParticleReal q,
                     const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz
                     const amrex::ParticleReal gBx, const amrex::ParticleReal gBy, const amrex::ParticleReal gBz)
{
    using namespace amrex::literals;

    amrex::ParticleReal const u2 = ux*ux + uy*uy + uz*uz;

    // Compute |B|
    const amrex::ParticleReal Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);

    // Limit Bmag to threshold value Bmin to avoid division by zero
    constexpr amrex::ParticleReal Bmin  = 1e-12_prt;     // Tesla
    //constexpr amrex::ParticleReal B2min = Bmin * Bmin;
    const amrex::Real invB = 1.0_prt / amrex::max(Bmag, Bmin);

    // 2. Compute inverse Lorentz factor
    const Real u2 = static_cast<Real>(ux*ux + uy*uy + uz*uz);
    constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + u2*inv_c2);

    // Compute blending coefficient alpha
    const amrex::ParticleReal alpha = ComputeBlendedAlpha(Bx, By, Bz, q, m, dt);

    // If B is below threshold, fall back to standard x += v dt
    // otherwise do standard blended position update with drifts
    if (Bmag <= Bmin) {

#if !defined(WARPX_DIM_1D_Z)
        x += ux * inv_gamma * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
        y += uy * inv_gamma * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        z += uz * inv_gamma * dt;
#endif
        return;
    }

    //define unit vector components
    const ParticleReal bx = Bx * invB;
    const ParticleReal by = By * invB;
    const ParticleReal bz = Bz * invB;

    //compute parallel and perpendicular momenta
    const amrex::ParticleReal upar_x = ux * bx;
    const amrex::ParticleReal upar_y = uy * by;
    const amrex::ParticleReal upar_z = uz * bz;

    const amrex::ParticleReal uperp_x = ux - ux*bx;
    const amrex::ParticleReal uperp_y = uy - uy*by;
    const amrex::ParticleReal uperp_z = uz - uz*bz;

    //compute magnetic moment
    const amrex::ParticleReal mu = m*inv_gamma*(uperp_x*uperp_x + uperp_y*uperp_y + uperp_z*uperp_z) *0.5_prt*invB;

    // E×B drift: v_E = (c E × B) / B^2
    const Real ExB_x = Ey*Bz - Ez*By;
    const Real ExB_y = Ez*Bx - Ex*Bz;
    const Real ExB_z = Ex*By - Ey*Bx;

    const Real vE_x =  ExB_x * invB * invB;
    const Real vE_y =  ExB_y * invB * invB;
    const Real vE_z =  ExB_z * invB * invB;

    // Grad-B drift: v_gradB = (mu / (q B)) (B × ∇B) / B^2
    const Real bXgradB_x = by*gBz - bz*gBy;
    const Real bXgradB_y = bz*gBx - bx*gBz;
    const Real bXgradB_z = bx*gBy - by*gBx;

    const Real vgradB_x = (mu * invB * invB / q) * bXgradB_x;
    const Real vgradB_y = (mu * invB * invB / q) * bXgradB_y;
    const Real vgradB_z = (mu * invB * invB / q) * bXgradB_z;


    // Curvature drift: v_c = (m gamma u_par^2) / (q B) (b × (b · ∇) b) / B
    const Real b_dot_gradb_x = bx*gBx + by*gBy + bz*gBz;
    const Real b_dot_gradb_y = by*gBx + by*gBy + bz*gBz;
    const Real b_dot_gradb_z = bz*gBx + by*gBy + bz*gBz;

    const Real bXbdotgradb_x = by*b_dot_gradb_z - bz*b_dot_gradb_y;
    const Real bXbdotgradb_y = bz*b_dot_gradb_x - bx*b_dot_gradb_z;
    const Real bXbdotgradb_z = bx*b_dot_gradb_y - by*b_dot_gradb_x;

    //
    vcurv_x = (m * (inv_gamma*upar_x)*(inv_gamma*upar_x)  * invB / q) * bXbdotgradb_x;
    vcurv_y = (m * inv_gamma
    


    // massive particles
    if (mass > 0.0_rt) {
        constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
        const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + u2*inv_c2);

        // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
        x += ux * inv_gamma * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
        y += uy * inv_gamma * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        z += uz * inv_gamma * dt;
#endif
    }
    // massless particles (photons)
    else {
        if (u2 > 0._prt) {
            amrex::ParticleReal const c_over_unorm = PhysConst::c/std::sqrt(u2);

            // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
            x += ux * c_over_unorm * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
            y += uy * c_over_unorm * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
            z += uz * c_over_unorm * dt;
#endif
        }
    }
}

/** \brief Compute the inverse Lorentz factor for the position update in
 *         the implicit methods, which is the average of gamma at time
 *         levels n and n+1.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal GetImplicitGammaInverse (const amrex::ParticleReal uxp_n,
                                             const amrex::ParticleReal uyp_n,
                                             const amrex::ParticleReal uzp_n,
                                             const amrex::ParticleReal uxp_nph,
                                             const amrex::ParticleReal uyp_nph,
                                             const amrex::ParticleReal uzp_nph) noexcept
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal inv_c2 = 1.0_prt/(PhysConst::c*PhysConst::c);

    const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph - uxp_n;
    const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph - uyp_n;
    const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph - uzp_n;
    const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n*uxp_n + uyp_n*uyp_n + uzp_n*uzp_n)*inv_c2);
    const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
    const amrex::ParticleReal gaminv = 2._prt/(gamma_n + gamma_np1);

    return gaminv;
}

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz`.
 *    The implicit version is the Crank-Nicolson scheme,
 *    x^{n+1} - x^{n} = dt*(u^{n+1} + u^{n})/(gamma^{n+1} + gamma^{n})
 *    See Eqs. 15 and 17 in Chen, JCP 407 (2020) 109228.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePositionImplicit ([[maybe_unused]] amrex::ParticleReal& x,
                             [[maybe_unused]] amrex::ParticleReal& y,
                             [[maybe_unused]] amrex::ParticleReal& z,
                             const amrex::ParticleReal ux_n, const amrex::ParticleReal uy_n, const amrex::ParticleReal uz_n,
                             const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
                             const amrex::Real dt )
{

    // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
    const amrex::ParticleReal inv_gamma = GetImplicitGammaInverse(ux_n, uy_n, uz_n, ux, uy, uz);

    // Update positions over one time step
#if !defined(WARPX_DIM_1D_Z)
    x += ux * inv_gamma * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Radial pushes particles in 3D
    y += uy * inv_gamma * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
    z += uz * inv_gamma * dt;
#endif
}

/** \brief Check particle position for convergence. This is used by the Picard method
 *         used to achieve a self-consistent time-centered update of the particles
 *         for given electric and magnetic fields on the grid.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void PositionNorm ([[maybe_unused]] const amrex::ParticleReal dxp,
                   [[maybe_unused]] const amrex::ParticleReal dyp,
                   [[maybe_unused]] const amrex::ParticleReal dzp,
                   [[maybe_unused]] const amrex::ParticleReal dxp_save,
                   [[maybe_unused]] const amrex::ParticleReal dyp_save,
                   [[maybe_unused]] const amrex::ParticleReal dzp_save,
                   [[maybe_unused]] const amrex::ParticleReal idxg2,
                   [[maybe_unused]] const amrex::ParticleReal idyg2,
                   [[maybe_unused]] const amrex::ParticleReal idzg2,
                   amrex::ParticleReal& step_norm )
{
    using namespace amrex::literals;

#if !defined(WARPX_DIM_RCYLINDER)
    step_norm = (dzp - dzp_save)*(dzp - dzp_save)*idzg2;
#else
    step_norm = 0._prt;
#endif
#if !defined(WARPX_DIM_1D_Z)
    step_norm += (dxp - dxp_save)*(dxp - dxp_save)*idxg2;
#endif
#if defined(WARPX_DIM_3D)
    step_norm += (dyp - dyp_save)*(dyp - dyp_save)*idyg2;
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    step_norm += (dyp - dyp_save)*(dyp - dyp_save)*idxg2;
#endif
    step_norm = std::sqrt(step_norm);

}

#endif // WARPX_PARTICLES_PUSHER_UPDATEPOSITION_H_
