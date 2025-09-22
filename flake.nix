{
  description = "torch-image-metrics development environment";

  inputs = {
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils, ... }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "torch-image-metrics-dev";
      
          buildInputs = [
            # Python interpreter with venv module for uv
            pkgs.python310
            pkgs.uv

            # System dependencies for PyTorch and image processing
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.libGL
            pkgs.glib
            pkgs.wget
            pkgs.cudatoolkit  # For CUDA support (optional)
            pkgs.graphviz     # For documentation generation
            
            # Additional development tools
            pkgs.git
          ];
          
          env = {
            # Required for dynamic linking
            LD_LIBRARY_PATH = "${
              with pkgs;
              lib.makeLibraryPath [
                zlib
                stdenv.cc.cc.lib
                libGL
                glib
              ]
            }:/run/opengl-driver/lib";
          };

          shellHook = ''
            # Set CUDA path if available
            export CUDA_PATH=${pkgs.cudatoolkit}
            
            # Initialize uv virtual environment if not exists
            if [ ! -d ".venv" ]; then
              echo "Creating uv virtual environment..."
              uv venv
            fi
            
            # Activate virtual environment
            source .venv/bin/activate
            
            # Sync dependencies
            echo "Syncing dependencies..."
            uv sync
            
            echo "torch-image-metrics development environment ready!"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
          '';
        };
      }
    );
}