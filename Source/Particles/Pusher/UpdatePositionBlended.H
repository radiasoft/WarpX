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
//#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include "Particles/Pusher/ComputeBlendedAlpha.H"


/** \brief Push the particle position over one time step,
 *         given the value of its momenta `ux`, `uy`, `uz`,
 *         magnetic field components, `Bx`, `By`, `Bz`,
 *         and magnetic field gradients `gradBx`, `gradBy`, `gradBz`,
 *         using the Blended position update, including estimated drifts (gradB, ExB, curvature)
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdatePositionBlended ( amrex::ParticleReal& x, amrex::ParticleReal& y, amrex::ParticleReal& z,
                     const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz,
                     const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
                     const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
                     const amrex::ParticleReal gradBx, const amrex::ParticleReal gradBy, const amrex::ParticleReal gradBz,
                     const amrex::ParticleReal kappax, const amrex::ParticleReal kappay, const amrex::ParticleReal kappaz,
                     const amrex::Real dt,
                     amrex::ParticleReal const m, const amrex::ParticleReal q)

{
    using namespace amrex::literals;

    // Compute blending coefficient alpha
    const amrex::ParticleReal alpha = ComputeBlendedAlpha(Bx, By, Bz, q, m, dt);

    amrex::ParticleReal const u2 = ux*ux + uy*uy + uz*uz;

    // massive particles
    if (m > 0.0_rt) {
        
        constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
        const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + u2*inv_c2);


        // Compute |B|
        const amrex::ParticleReal Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);

        // Limit Bmag to threshold value Bmin to avoid division by zero
        constexpr amrex::ParticleReal Bmin  = 1e-12_prt;     // Tesla
        //constexpr amrex::ParticleReal B2min = Bmin * Bmin;
        const amrex::Real invB = 1.0_prt / amrex::max(Bmag, Bmin);

        // Compute blending coefficient alpha
        //const amrex::ParticleReal alpha = ComputeBlendedAlpha(Bx, By, Bz, q, m, dt);

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
        const amrex::ParticleReal bx = Bx * invB;
        const amrex::ParticleReal by = By * invB;
        const amrex::ParticleReal bz = Bz * invB;

        //compute parallel and perpendicular momenta
        const amrex::ParticleReal u_dot_b = ux*bx + uy*by + uz*bz;
        const amrex::ParticleReal upar_x  = u_dot_b * bx;
        const amrex::ParticleReal upar_y  = u_dot_b * by;
        const amrex::ParticleReal upar_z  = u_dot_b * bz;

        const amrex::ParticleReal uperp_x = ux - upar_x;
        const amrex::ParticleReal uperp_y = uy - upar_y;
        const amrex::ParticleReal uperp_z = uz - upar_z;

        //compute magnetic moment
        const amrex::ParticleReal mu = m*inv_gamma*(uperp_x*uperp_x + uperp_y*uperp_y + uperp_z*uperp_z) *0.5_prt*invB;

        // E×B drift: v_E = (c E × B) / B^2
        const amrex::ParticleReal ExB_x = Ey*Bz - Ez*By;
        const amrex::ParticleReal ExB_y = Ez*Bx - Ex*Bz;
        const amrex::ParticleReal ExB_z = Ex*By - Ey*Bx;

        const amrex::ParticleReal vE_x =  PhysConst::c * ExB_x * invB * invB;
        const amrex::ParticleReal vE_y =  PhysConst::c * ExB_y * invB * invB;
        const amrex::ParticleReal vE_z =  PhysConst::c * ExB_z * invB * invB;

        // Grad-B drift: v_gradB = (mu / (q B)) (B × ∇B) / B^2
        const amrex::ParticleReal bXgradB_x = By*gradBz - Bz*gradBy;
        const amrex::ParticleReal bXgradB_y = Bz*gradBx - Bx*gradBz;
        const amrex::ParticleReal bXgradB_z = Bx*gradBy - By*gradBx;

        const amrex::ParticleReal vgradB_x = (mu * invB * invB / q) * bXgradB_x;
        const amrex::ParticleReal vgradB_y = (mu * invB * invB / q) * bXgradB_y;
        const amrex::ParticleReal vgradB_z = (mu * invB * invB / q) * bXgradB_z;


        // Curvature drift: v_c = (γ^2 v_par^2)/((q/m) * Bmag) * (b × kappa)

        const amrex::ParticleReal bXkx = by*kappaz - bz*kappay;
        const amrex::ParticleReal bXky = bz*kappax - bx*kappaz;
        const amrex::ParticleReal bXkz = bx*kappay - by*kappax;

        const amrex::ParticleReal coeff = m*(upar_x*upar_x + upar_y*upar_y + upar_z*upar_z) * invB / q;

        // vcurv_x = (m * (inv_gamma*upar_x)*(inv_gamma*upar_x)  * invB / q * bXkx;
        const amrex::ParticleReal vcurv_x = coeff * bXkx;
        const amrex::ParticleReal vcurv_y = coeff * bXky;
        const amrex::ParticleReal vcurv_z = coeff * bXkz;


        // net drift velocity
        const amrex::ParticleReal vx_drift = vE_x + vgradB_x + vcurv_x;
        const amrex::ParticleReal vy_drift = vE_y + vgradB_y + vcurv_y;
        const amrex::ParticleReal vz_drift = vE_z + vgradB_z + vcurv_z;

        // Update positions over one time step
        // Blended update blends perpendicular and drift motion via alpha
#if !defined(WARPX_DIM_1D_Z)
            x += upar_x * inv_gamma * dt + alpha * uperp_x inv_gamma * dt + (1._prt - alpha) * vx_drift * dt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Particles pushed in 3D even in 1D cylindrical, 1D spherical, and 2D cylindrical geometry
            y += upar_y * inv_gamma * dt + alpha * uperp_y * dt + (1._prt - alpha) * vy_drift * dt;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
            z += upar_z * inv_gamma * dt + alpha * uperp_z * dt + (1._prt - alpha) * vz_drift * dt;
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

#endif // WARPX_PARTICLES_PUSHER_UPDATEPOSITIONBLENDED_H_
