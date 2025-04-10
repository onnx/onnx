set_toolchains("cross")
set_plat("windows")
set_arch("arm64")

-- Define the ONNX target
target("onnx_arm64")
    set_kind("binary") -- Use "binary" for executables or "shared" for shared libraries
    add_files("src/*.cpp") -- Add source files
    add_includedirs("include") -- Add include directories
    add_links("protobuf", "onnx_proto") -- Link required libraries

    -- Custom build step to create Python wheel
    on_build(function (target)
        -- Ensure Python dependencies are installed
        os.exec("python -m pip install --upgrade pip setuptools wheel")

        -- Run the Python build process
        os.exec("python -m build --wheel")
    end)