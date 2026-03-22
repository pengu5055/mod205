let
  pkgs = import <nixpkgs> { config.allowUnfree = true; };
  py = pkgs.python310;
  pyPkgs = pkgs.python310Packages;
in
pkgs.mkShell {
  shellHook = ''
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:${pkgs.zlib}/lib:${pkgs.blas}/lib:${pkgs.lapack}/lib:$LD_LIBRARY_PATH"
    if [ -n "$PS1" ] && [ -z "$FISH_VERSION" ]; then
      export SHELL=${pkgs.fish}/bin/fish
      ${pkgs.fish}/bin/fish --login
    fi
    echo "LD_LIBRARY_PATH in devShell: $LD_LIBRARY_PATH"
  '';
  buildInputs =
    (with pkgs; [
      git
      ffmpeg
      stdenv.cc.cc
      stdenv.cc.cc.lib
      qt6.qtbase
      qt6.qtwayland
      openmpi
    ]) ++
    (with pyPkgs; [
      pip
      setuptools
      wheel
      tkinter
      mpi4py
      # h5py
      # h5py-mpi
      # hdf5plugin
      # pyqt6  # -> Enable if HiDPI Display, GNOME likes Qt more than tk
    ]) ++ [
      py
    ];
}
