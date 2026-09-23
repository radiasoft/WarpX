# Add tests (alphabetical order) ##############################################
add_warpx_test(
    test_3d_ionization_rudd  # name
    3  # dims
    1  # nprocs; also runs in non-MPI builds
    inputs_test_3d_ionization_rudd  # inputs
    "analysis_ionization_rudd.py diags/diag1000001"  # analysis
    OFF  # checksum: stochastic physics checks; baseline pending a reference run
    OFF  # dependency
)

# Cross sections are opened relative to the test's working directory.
if(TEST test_3d_ionization_rudd.run)
    configure_file(ionization.dat
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_3d_ionization_rudd/ionization.dat COPYONLY)
endif()
