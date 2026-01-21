{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default =
        with pkgs;
        mkShell {
          packages = [
            bashInteractive
            python313
            python313Packages.venvShellHook
          ];

          venvDir = ".venv";

          env.LD_LIBRARY_PATH =
            with pkgs;
            lib.makeLibraryPath [
              stdenv.cc.cc.lib
              zlib
            ];
        };
    };
}
