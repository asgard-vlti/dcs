{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    self.submodules = true;
  };

  outputs = inputs@{ flake-parts, yorick-flake, shmimshow, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [

      ];
      systems = [ "x86_64-linux" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        packages = rec {
          baldr = pkgs.stdenv.mkDerivation {
            name = "baldr";
            src = ./.;
            buildInputs = with pkgs; [ 
              cmake nlohmann_json pkg-config
              boost cppzmq fmt fftw tomlplusplus
              cfitsio libb64
            ];
            preBuild = ''
              ls
              find \
                    -name CMakeCache.txt \
                    -exec rm {} \;
              export CPATH=$CPATH:${pkgs.eigen}/include/eigen3:${pkgs.tomlplusplus}/include/toml++
            '';
            installPhase = ''
              cp -r baldr_jcr/baldr $out
            '';
          };

          default = baldr;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            boost
            catch2
            cmake
            cfitsio
            cppzmq
            tomlplusplus
            eigen
            nlohmann_json
            fmt
            fftw
            pkg-config
            libb64
          ];
          shellHook = ''
            source .venv/bin/activate
            export CPATH=$CPATH:${pkgs.eigen}/include/eigen3:${pkgs.tomlplusplus}/include/toml++
            alias makecomstructs="uv run $(pwd)/make_commander_structs.py $(pwd)/baldr_jcr/baldr.h $(pwd)/baldr_jcr/commander_structs.h";
          '';
        };
      };
    };
}
