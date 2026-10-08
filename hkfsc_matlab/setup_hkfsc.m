function setup_hkfsc()
%SETUP_HKFSC Add this standalone package root to the MATLAB path.
root = fileparts(mfilename('fullpath'));
addpath(root);
end
