function net = resDNet_joint_train(net,varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_layers', 'vl_setupnn.m')) ;

%  ---- Network Training Parameters ------
opts.numEpochs = 300; % Maximum number of gradient-descent iterations
opts.batchSize = 50;
opts.batchSize_val = 100;
opts.learningRate = 0.001;
opts.solver = @solver.momentum;
opts.solverOpts = [];%struct('gamma', 0.9, 'decay', 0);
opts.cudnn = true;
opts.backPropDepth = +inf;
opts.net_move = @net_move;
opts.net_eval = @resDNet_eval;

%Indicate whether to use or not a gpu for the training.
opts.gpus = 1; %[] -> no gpu
opts.plotStatistics = false; % if set to true the graph with the objective is ploted
opts.saveFreq = 10; % Results are saved every opts.saveFreq number of epochs.
opts.conserveMemory = true;

opts.resnetLayers = 1;
opts.imdbPath = fullfile('..','data', 'imdb.mat');
opts.patchFile = []; % Mat file that contains the Nbrs_idx and dist 
% for all the training and testing data.

opts.intScale = 1; % A scalar that scales the intensity of the data. 
% If set to 255, then the dynamic range of the images will be in [0,1].

[opts,varargin] = vl_argparse(opts,varargin);


opts.noise_std = [5,9,13,17,21,25,29];
%  ------ Inverse Problem Parameters ------
% Define the forward model of the inverse problem we are training the
% network to solve.
% y=A(x)+n where A is the forward operator and n is gaussian noise of
% noise_std.

opts.name_id = 'ResDNet';
opts.randn_seed = 124324;

% layer-specific paramaters
%  ----- Resnet -------
opts.hdims = [3,3,1,64]; % Dimensions of the filterbank
opts.h = [];
opts.b = [];
opts.s = [];
opts.padSize=[];
opts.zeroMeanFilters_h = true;
opts.weightNormalization_h = true;
opts.init_h = 'dct';
opts.lr_h = [1,1,1];

opts.g = [];
opts.sg = [];
opts.weightNormalization_g = true;
opts.Nbrs = 8;

opts.hdims2 = [3,3,64,64]; % Dimensions of the filterbank in the inner Layer
opts.h2 = [];
opts.b2 = [];
opts.s2 = [];
opts.padSize2=[];
opts.zeroMeanFilters_h2 = false;
opts.weightNormalization_h2 = false;
opts.init_h2 = 'dct';
opts.lr_h2 = [1,1,1];

opts.stride=[1,1];
opts.dilate = 1;
opts.padType='symmetric';

opts.fdims = [3,3,64,64];
opts.rh = [];
opts.rb = [];
opts.rs = [];
opts.pa_lb = 0;
opts.pa_ub = 200;
opts.pa_a = 0.1;

opts.fdims2 = [3,3,64,64];
opts.rh2 = [];
opts.rb2 = [];
opts.rs2 = [];
opts.pa_lb2 = 0;
opts.pa_ub2 = 200;
opts.pa_a2 = 0.1;
opts.zeroMeanFilters = false(1,2);
opts.weightNormalization = false(1,2);
opts.shortcut = true;
opts.resPreActivationInit = 'dct';% msra
opts.lr_res = 1;

opts.lr = 1;
opts.resPreActivationType = 'prelu';
opts.activation = 'clamp';
opts.lb = 0;
opts.ub = inf;
opts.inLayer = false;

% --- L2Proj layer -----
opts.alpha = 0;
% --- Imloss layer -----
opts.peakVal = 255;
opts.loss_type = 'psnr';
% --- CLAMP layer -----
opts.clb = 0;
opts.cub = 255;
opts.lastLayer = false;
% ---- PRELU layer ----
opts.prelu_a = 0.1;
% ---- PCLAMP layer ---
opts.pclamp_lb = 0;
opts.pclamp_ub = 200;

opts.cid = 'single';
%opts.cid='double';

% Patch-Match parameters
opts.BlockMatch.searchwin = [15 15];
opts.BlockMatch.useSep = true;
opts.BlockMatch.transform = false;
opts.BlockMatch.patchDist = 'euclidean';
opts.BlockMatch.sorted = true;
opts.BlockMatch.Wx = [];
opts.BlockMatch.Wy = [];
opts.BlockMatch.multichannel_dist = false; % If set to true the computation of the 
% distance between patches is performed using the multichannel patch. 
% Otherwise it is computed using a single channel patch, which is derived 
% as the mean of the image channels. 
opts.NonLocal = false;

[opts,varargin] = vl_argparse(opts, varargin);

resLayer = ['resPreActivationLayer_' opts.resPreActivationType];
if opts.NonLocal 
  WLayer = 'pgtcf'; WTLayer = 'pgtcft'; 
  opts.lr_h = ones(1,5); opts.lr_h2 = ones(1,5);
else
  WLayer = 'conv2D'; WTLayer = 'conv2Dt'; 
end

opts.net_struct={...
  struct('layer_type',WLayer), ...
  struct('layer_type', opts.activation), ...
  struct('layer_type','conv2D','inLayer',true), ...
  struct('layer_type',resLayer), ...
  struct('layer_type',WTLayer), ...
  struct('layer_type', 'l2Proj'), ...
  struct('layer_type', 'subs'), ...
  struct('layer_type','clamp','lastLayer',true),...
  struct('layer_type','imloss')};

opts.net_struct = [opts.net_struct(1:3) ...
  repmat(opts.net_struct(4),1,opts.resnetLayers) opts.net_struct(5:end)];

% opts.net_struct={...
%   struct('layer_type',WLayer), ...
%   struct('layer_type',resLayer,'shortcut',false), ...
%   struct('layer_type',resLayer), ...
%   struct('layer_type',resLayer), ...
%   struct('layer_type',WTLayer), ...
%   struct('layer_type', 'l2Proj'), ...
%   struct('layer_type', 'subs'), ...
%   struct('layer_type','clamp','lastLayer',true),...
%   struct('layer_type','imloss')};



% opts.net_struct={...
%   struct('layer_type','conv2D','learningRate',[1,1,1], ...
%   'weightNormalization',true,'zeroMeanFilters',true), ...
%   struct('layer_type', 'clamp', 'lb', 0, 'ub', inf), ...
%   struct('layer_type','conv2D','learningRate',[1,1,1], 'hdims', ...
%   [3,3,64,64],'weightNormalization',true,'zeroMeanFilters',true), ...
%   struct('layer_type','resBlock2','learningRate',ones(1,6), ...
%   'weightNormalization',true(1,2), 'zeroMeanFilters',true(1,2)), ...
%   struct('layer_type','conv2Dt','learningRate',[1,1,1], ...
%   'weightNormalization',true,'zeroMeanFilters',true), ...
%   struct('layer_type', 'l2Proj', 'alpha', 0), ...
%   struct('layer_type', 'subs'), ...
%   struct('layer_type','clamp'),...
%   struct('layer_type','imloss')};
% 
% opts.net_struct = [opts.net_struct(1:3) ...
%   repmat(opts.net_struct(4),1,opts.resnetLayers) opts.net_struct(5:end)];
% opts.net_struct{4}.first_resLayer = opts.first_resLayer;
    

[opts,varargin] = vl_argparse(opts, varargin);

if isempty(opts.solverOpts)
  opts.solverOpts = opts.solver();
end

opts.netParams = struct();

% How many residual blocks the network consists of.
if isempty(net)
  numStages = 0;
  for k=1:numel(opts.net_struct)
    switch opts.net_struct{k}.layer_type
      case{'resPreActivationLayer_relu','resPreActivationLayer_prelu', ...
          'resPreActivationLayer_pclamp'}
        numStages = numStages+1;
    end
  end
else
  numStages = 0;
  for k=1:numel(net.layers)
    switch net.layers{k}.type
      case{'resPreActivationLayer_relu','resPreActivationLayer_prelu', ...
          'resPreActivationLayer_pclamp'}
        numStages = numStages+1;
    end
  end
end

str_solver = char(opts.solver);
str_solver = str_solver(strfind(str_solver,'.')+1:end);

if numel(opts.noise_std) == 1
  str_std = sprintf('%0.f',opts.noise_std(1));
else
  str_std = [ '[' num2str(opts.noise_std) ']' ];
  str_std = regexprep(str_std,'\s*',',');
end

if isempty(net)
  if isfield(opts.net_struct{1},'hdims')
    hdims = opts.net_struct{1}.hdims;
  else
    hdims = opts.hdims;
  end
  if isfield(opts.net_struct{3},'fdims')
    fdims = opts.net_struct{3}.fdims;
  else
    fdims = opts.fdims;
  end  
else
  hdims = size(net.layers{1}.weights{1});
  fdims = size(net.layers{3}.weights{1});
end

opts.expDir = fullfile('Results', ...
  sprintf(['%s-stages:%.0f-conv:%.0fx%.0fx%.0f@%.0f-res:' ...
    '%.0fx%.0fx%.0f@%.0f-std:%s-solver:%s-jointTrain'],...
    opts.name_id,numStages,hdims(1),hdims(2),hdims(3),hdims(4), ...
    fdims(1),fdims(2),fdims(3),fdims(4),str_std,str_solver));

if ~exist(opts.expDir, 'dir')
  mkdir(opts.expDir);
  copyfile([mfilename '.m'], [opts.expDir filesep]);
  save([opts.expDir filesep 'arg_In'],'opts');
end

opts = vl_argparse(opts, varargin);

% -------------------------------------------------------------------------
%                Prepare Data and Model
% -------------------------------------------------------------------------
imdb = load(opts.imdbPath);
if ~isequal(opts.cid,'single')
  imdb.images.data = cast(imdb.images.data,opts.cid);
end

imdb.images.data = imdb.images.data/opts.intScale;

% imdb.images.data = imdb.images.data(:,:,:,313:321);% 8:10
% imdb.images.set = imdb.images.set(313:321);

noise_levels = numel(opts.noise_std);
imdb.images.set = imdb.images.set(:);
imdb.images.set = repmat(imdb.images.set,noise_levels,1); % Every image is 
% corrupted by K different noise levels and all the K instances are used 
% either for training or for validation.


% Create the noise added to the data according to the chosen standard
% deviation of the noise.

% Initialize the seed for the random generator
s = RandStream('mt19937ar','Seed',opts.randn_seed);
RandStream.setGlobalStream(s);

% The degraded input that we feed to the network and we want to
% reconstruct.
imdb.images.obs = [];
for k=1:noise_levels
  imdb.images.obs = cat(4,imdb.images.obs, imdb.images.data + ...
  (opts.noise_std(k)/opts.intScale)*randn(size(imdb.images.data),opts.cid));
end
imdb.images.noise_std = opts.noise_std/opts.intScale;
imdb.images.stage_input = [];

opts.inputSize = size(imdb.images.data(:,:,:,1));
% Initialize network parameters
% Initialize network parameters
if isempty(net)
  net = net_init_from_struct(opts);
else
  net.meta.trainOpts.inputSize = opts.inputSize;
  net.meta.trainOpts.noise_std = opts.noise_std;
  net.meta.trainOpts.randSeed = opts.randn_seed;
  net.meta.trainOpts.numEpochs = opts.numEpochs;
  net.meta.trainOpts.learningRate = opts.learningRate;
  net.meta.trainOpts.optimizer = char(opts.solver);
  net.meta.netParams = opts.netParams;
end

if strcmp(net.layers{1}.type,'pgtcf')
  
  net.meta.BlockMatchParams = opts.BlockMatch;
% -------------------------------------------------------------------------
% Compute patch-similarity based either on the noisy input or in a initial
% denoised estimate and use the same similarity indices for all stages.
% -------------------------------------------------------------------------
  if isempty(opts.patchFile)
    opts.patchFile = fullfile(opts.expDir,'patchSimilarityIndices.mat');
  end
  opts_PM = struct('searchwin',opts.BlockMatch.searchwin,'useSep', ...
    opts.BlockMatch.useSep, 'transform',opts.BlockMatch.transform, ...
    'patchDist',opts.BlockMatch.patchDist,'sorted', ...
    opts.BlockMatch.sorted,'Wx',opts.BlockMatch.Wx,'Wy', ...
    opts.BlockMatch.Wy,'Nbrs',opts.Nbrs,'cudnn', ...
    opts.cudnn, 'padSize', net.layers{1}.padSize, 'padType', ...
    net.layers{1}.padType,'gpus',opts.gpus,'patchSize', ...
    hdims(1:2),'stride',net.layers{1}.stride,'multichannel_dist', ...
    opts.BlockMatch.multichannel_dist,'batchSize',opts.batchSize_val);
  
  if exist(opts.patchFile, 'file')
    load(opts.patchFile,'Nbrs_idx');
  else
    [Nbrs_idx,Nbrs_dist] = precompute_knn_idx(imdb.images.obs,opts_PM); %#ok<ASGLU>
    save(opts.patchFile,'Nbrs_idx','Nbrs_dist','-v7.3');
    clear Nbrs_dist;
  end
else
  Nbrs_idx = [];
end

% -------------------------------------------------------------------------
%                     Train Network Stage by Stage
% -------------------------------------------------------------------------

train_image_set = find(imdb.images.set == 1);
val_image_set = find(imdb.images.set == 2);

start_time = tic;
net = deep_net_train(net, imdb, @(x,y)getBatch(x,y,Nbrs_idx), ...
  'expDir', opts.expDir, ...
  'solver', opts.solver, ...
  'solverOpts', opts.solverOpts, ...
  'batchSize', opts.batchSize, ...
  'gpus', opts.gpus,...
  'train', train_image_set, ...
  'val', val_image_set, ...
  'numEpochs', opts.numEpochs, ...
  'learningRate', opts.learningRate, ...
  'conserveMemory', opts.conserveMemory, ...
  'backPropDepth', opts.backPropDepth, ...
  'cudnn', opts.cudnn, ...
  'saveFreq', opts.saveFreq, ...
  'plotStatistics', opts.plotStatistics, ...
  'netParams', opts.netParams, ...
  'net_move', opts.net_move, ...
  'net_eval', opts.net_eval);

train_time = toc(start_time);
fprintf('\n-------------------------------------------------------\n')
fprintf('\n\n The training was completed in %.2f secs.\n\n',train_time);
fprintf('-------------------------------------------------------\n')

save(fullfile(opts.expDir,'net-final.mat'), 'net');



function [im, im_gt, aux] = getBatch(imdb,batch,Nbrs_idx)

N_gt = size(imdb.images.data,4);% Number of unique ground-truth images used 
% for training / testing.
im_gt = imdb.images.data(:,:,:,mod(batch-1,N_gt)+1);
im = imdb.images.obs(:,:,:,batch);

% Instead of using a vector noise_std of size N_gt*K (K : number of
% the different noise levels and N_gt the number of unique ground-truth 
% images) we use only a vector of size K. Then the first 1:N_gt images in 
% the data set are distorted by noise with standard deviation equal to 
% noise_std(1), the next N_gt+1:2*N_gt by noise with standard deviation 
% equal to im_noise_std(2), etc.

if isempty(Nbrs_idx)
  idx = [];
else
  idx = Nbrs_idx(:,:,:,batch);
end

aux = struct('stdn',imdb.images.noise_std(ceil(batch/N_gt)), ...
             'Nbrs_idx',idx);


% -------------------------------------------------------------------------
%                       Initialize Network
% -------------------------------------------------------------------------

function net = net_init_from_struct(opts)

net_add_layer = @resDNet_add_layer;
net.layers = {};
num_layers = numel(opts.net_struct);

for l = 1:num_layers
  
  switch opts.net_struct{l}.layer_type
    
    case {'conv2D','conv2Dt','pgtcf','pgtcft'}
      if ~isfield(opts.net_struct{l},'inLayer')
        opts.net_struct{l}.inLayer = opts.inLayer;
      end
      if ~opts.net_struct{l}.inLayer       
        if ~isfield(opts.net_struct{l},'hdims')
          opts.net_struct{l}.hdims = opts.hdims;
        end
        if ~isfield(opts.net_struct{l},'h')
          opts.net_struct{l}.h = opts.h;
        end
        if ~isfield(opts.net_struct{l},'b')
          opts.net_struct{l}.b = opts.b;
        end
        if ~isfield(opts.net_struct{l},'s')
          opts.net_struct{l}.s = opts.s;
        end
        if ~isfield(opts.net_struct{l},'padSize')
          opts.net_struct{l}.padSize = opts.padSize;
        end
        if ~isfield(opts.net_struct{l},'zeroMeanFilters_h')
          opts.net_struct{l}.zeroMeanFilters_h = opts.zeroMeanFilters_h;
        end
        if ~isfield(opts.net_struct{l},'weightNormalization_h')
          opts.net_struct{l}.weightNormalization_h = opts.weightNormalization_h;
        end
        if ~isfield(opts.net_struct{l},'learningRate')
          opts.net_struct{l}.learningRate = opts.lr_h;
        end
        if ~isfield(opts.net_struct{l},'initType')
          opts.net_struct{l}.initType = opts.init_h;
        end        
      else
        if ~isfield(opts.net_struct{l},'hdims')
          opts.net_struct{l}.hdims = opts.hdims2;
        end
        if ~isfield(opts.net_struct{l},'h')
          opts.net_struct{l}.h = opts.h2;
        end
        if ~isfield(opts.net_struct{l},'b')
          opts.net_struct{l}.b = opts.b2;
        end
        if ~isfield(opts.net_struct{l},'s')
          opts.net_struct{l}.s = opts.s2;
        end
        if ~isfield(opts.net_struct{l},'padSize')
          opts.net_struct{l}.padSize = opts.padSize2;
        end
        if ~isfield(opts.net_struct{l},'zeroMeanFilters_h')
          opts.net_struct{l}.zeroMeanFilters_h = opts.zeroMeanFilters_h2;
        end
        if ~isfield(opts.net_struct{l},'weightNormalization_h')
          opts.net_struct{l}.weightNormalization_h = opts.weightNormalization_h2;
        end
        if ~isfield(opts.net_struct{l},'learningRate')
          opts.net_struct{l}.learningRate = opts.lr_h2;
        end
        if ~isfield(opts.net_struct{l},'initType')
          opts.net_struct{l}.initType = opts.init_h2;
        end    
      end
              
      if ~isfield(opts.net_struct{l},'stride')
        opts.net_struct{l}.stride = opts.stride;
      end
      if ~isfield(opts.net_struct{l},'dilate')
        opts.net_struct{l}.dilate = opts.dilate;
      end      
      if ~isfield(opts.net_struct{l},'padType')
        opts.net_struct{l}.padType = opts.padType;
      end      
      
      if ~isfield(opts.net_struct{l},'g')
        opts.net_struct{l}.g = opts.g;
      end
      if ~isfield(opts.net_struct{l},'sg')
        opts.net_struct{l}.sg = opts.sg;
      end
      if ~isfield(opts.net_struct{l},'Nbrs')
        opts.net_struct{l}.Nbrs = opts.Nbrs;
      end
      if ~isfield(opts.net_struct{l},'weightNormalization_g')
        opts.net_struct{l}.weightNormalization_g = opts.weightNormalization_g;
      end      


      net = net_add_layer(net, ...
        'layer_id', l, ...
        'inputSize', opts.inputSize, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'cid', opts.cid, ...
        'hdims',opts.net_struct{l}.hdims, ...
        'h', opts.net_struct{l}.h, ...
        'b', opts.net_struct{l}.b, ...
        's', opts.net_struct{l}.s, ...
        'padSize', opts.net_struct{l}.padSize, ...
        'zeroMeanFilters_h', opts.net_struct{l}.zeroMeanFilters_h, ...
        'weightNormalization_h', opts.net_struct{l}.weightNormalization_h, ...      
        'inLayer', opts.net_struct{l}.inLayer, ...
        'stride', opts.net_struct{l}.stride, ...        
        'padType', opts.net_struct{l}.padType, ...
        'dilate', opts.net_struct{l}.dilate, ...
        'learningRate', opts.net_struct{l}.learningRate, ...
        'initType', opts.net_struct{l}.initType, ...
        'g', opts.net_struct{l}.g, ...
        'sg', opts.net_struct{l}.sg, ...
        'Nbrs', opts.net_struct{l}.Nbrs, ...
        'weightNormalization_g', opts.net_struct{l}.weightNormalization_g);

    case {'resPreActivationLayer_relu','resPreActivationLayer_prelu', ...
        'resPreActivationLayer_pclamp'}
      
      if ~isfield(opts.net_struct{l},'fdims')
        opts.net_struct{l}.fdims = opts.fdims;
      end      
      if ~isfield(opts.net_struct{l},'h')
        opts.net_struct{l}.h = opts.rh;
      end
      if ~isfield(opts.net_struct{l},'b')
        opts.net_struct{l}.b = opts.rb;
      end
      if ~isfield(opts.net_struct{l},'s')
        opts.net_struct{l}.s = opts.rs;
      end            
      if ~isfield(opts.net_struct{l},'lb')
        opts.net_struct{l}.lb = opts.pa_lb;
      end
      if ~isfield(opts.net_struct{l},'ub')
        opts.net_struct{l}.ub = opts.pa_ub;
      end
      if ~isfield(opts.net_struct{l},'a')
        opts.net_struct{l}.a = opts.pa_a;
      end            
      if ~isfield(opts.net_struct{l},'fdims2')
        opts.net_struct{l}.fdims2 = opts.fdims2;
      end                  
      if ~isfield(opts.net_struct{l},'h2')
        opts.net_struct{l}.h2 = opts.rh2;
      end
      if ~isfield(opts.net_struct{l},'b2')
        opts.net_struct{l}.b2 = opts.rb2;
      end
      if ~isfield(opts.net_struct{l},'s2')
        opts.net_struct{l}.s2 = opts.rs2;
      end      
      if ~isfield(opts.net_struct{l},'lb2')
        opts.net_struct{l}.lb2 = opts.pa_lb2;
      end
      if ~isfield(opts.net_struct{l},'ub2')
        opts.net_struct{l}.ub2 = opts.pa_ub2;
      end
      if ~isfield(opts.net_struct{l},'a2')
        opts.net_struct{l}.a2 = opts.pa_a2;
      end                  
      if ~isfield(opts.net_struct{l},'stride')
        opts.net_struct{l}.stride = opts.stride;
      end
      if ~isfield(opts.net_struct{l},'padType')
        opts.net_struct{l}.padType = opts.padType;
      end      
      if ~isfield(opts.net_struct{l},'dilate')
        opts.net_struct{l}.dilate = opts.dilate;
      end     
      if ~isfield(opts.net_struct{l},'shortcut')
        opts.net_struct{l}.shortcut = opts.shortcut;
      end     
      if ~isfield(opts.net_struct{l},'zeroMeanFilters')
        opts.net_struct{l}.zeroMeanFilters = opts.zeroMeanFilters;
      end
      if ~isfield(opts.net_struct{l},'weightNormalization')
        opts.net_struct{l}.weightNormalization = opts.weightNormalization;
      end
      if ~isfield(opts.net_struct{l},'learningRate')
        opts.net_struct{l}.learningRate = opts.lr_res;
      end
      if ~isfield(opts.net_struct{l},'initType')
        opts.net_struct{l}.initType = opts.resPreActivationInit;
      end
      
      net = net_add_layer(net, ...
        'layer_id', l, ...
        'inputSize', opts.inputSize, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'cid', opts.cid, ...
        'fdims',opts.net_struct{l}.fdims, ...
        'h', opts.net_struct{l}.h, ...
        'b', opts.net_struct{l}.b, ...
        's', opts.net_struct{l}.s, ...
        'lb',opts.net_struct{l}.lb, ...
        'ub',opts.net_struct{l}.ub, ...        
        'a',opts.net_struct{l}.a, ...        
        'fdims2',opts.net_struct{l}.fdims2, ...
        'h2', opts.net_struct{l}.h2, ...
        'b2', opts.net_struct{l}.b2, ...
        's2', opts.net_struct{l}.s2, ...    
        'lb2',opts.net_struct{l}.lb2, ...
        'ub2',opts.net_struct{l}.ub2, ...        
        'a2',opts.net_struct{l}.a2, ...               
        'stride', opts.net_struct{l}.stride, ...
        'padType', opts.net_struct{l}.padType, ...
        'dilate', opts.net_struct{l}.dilate, ...
        'shortcut', opts.net_struct{l}.shortcut, ...
        'zeroMeanFilters', opts.net_struct{l}.zeroMeanFilters, ...
        'weightNormalization', opts.net_struct{l}.weightNormalization, ...
        'learningRate', opts.net_struct{l}.learningRate, ...
        'initType', opts.net_struct{l}.initType);      

    case 'clamp'
      if ~isfield(opts.net_struct{l},'lastLayer')
        opts.net_struct{l}.lastLayer = opts.lastLayer;
      end      
      if opts.net_struct{l}.lastLayer      
        if ~isfield(opts.net_struct{l},'lb')
          opts.net_struct{l}.lb = opts.clb;
        end
        if ~isfield(opts.net_struct{l},'ub')
          opts.net_struct{l}.ub = opts.cub;
        end
      else
        if ~isfield(opts.net_struct{l},'lb')
          opts.net_struct{l}.lb = opts.lb;
        end
        if ~isfield(opts.net_struct{l},'ub')
          opts.net_struct{l}.ub = opts.ub;
        end
      end        
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'lb',opts.net_struct{l}.lb, ...
        'ub',opts.net_struct{l}.ub);      
      
    case 'pclamp'
      if ~isfield(opts.net_struct{l},'lb')
        opts.net_struct{l}.lb = opts.pclamp_lb;
      end
      if ~isfield(opts.net_struct{l},'ub')
        opts.net_struct{l}.ub = opts.pclamp_ub;
      end
      if ~isfield(opts.net_struct{l},'learningRate')
        opts.net_struct{l}.learningRate = opts.lr;
      end      
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'lb',opts.net_struct{l}.lb, ...
        'ub',opts.net_struct{l}.ub, ...
        'learningRate', opts.net_struct{l}.learningRate);            
      
    case 'prelu'
      if ~isfield(opts.net_struct{l},'a')
        opts.net_struct{l}.a = opts.prelu_a;
      end
      if ~isfield(opts.net_struct{l},'learningRate')
        opts.net_struct{l}.learningRate = opts.lr;
      end      
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'a',opts.net_struct{l}.a, ...
        'learningRate', opts.net_struct{l}.learningRate);                  
    
    case 'subs'
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type);

    case 'l2Proj'
      if ~isfield(opts.net_struct{l},'learningRate')
        opts.net_struct{l}.learningRate = opts.lr;
      end
      if ~isfield(opts.net_struct{l},'alpha')
        opts.net_struct{l}.alpha = opts.alpha;
      end      
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'alpha', opts.net_struct{l}.alpha, ...
        'learningRate', opts.net_struct{l}.learningRate);      
      
    case 'imloss'
      if ~isfield(opts.net_struct{l},'peakVal')
        opts.net_struct{l}.peakVal = opts.peakVal;
      end
      if ~isfield(opts.net_struct{l},'loss_type')
        opts.net_struct{l}.loss_type = opts.loss_type;
      end
      net = net_add_layer(net,'layer_id',l, ...
        'layer_type', opts.net_struct{l}.layer_type, ...
        'peakVal',opts.net_struct{l}.peakVal, ...
        'loss_type',opts.net_struct{l}.loss_type);
  end
end

% Meta parameters
net.meta.trainOpts.inputSize = opts.inputSize;
net.meta.trainOpts.noise_std = opts.noise_std;
net.meta.trainOpts.randSeed = opts.randn_seed;
net.meta.trainOpts.numEpochs = opts.numEpochs;
net.meta.trainOpts.learningRate = opts.learningRate;
net.meta.trainOpts.optimizer = char(opts.solver);
net.meta.netParams = opts.netParams;


function [knn_idx,knn_D] = precompute_knn_idx(im_net_input,opts)

[Nx,Ny,~,NI] = size(im_net_input);
Nx = Nx + sum(opts.padSize(1:2));
Ny = Ny + sum(opts.padSize(3:4));

patchDims= max(floor(([Nx,Ny]-opts.patchSize)./opts.stride)+1,0);

cid = class(im_net_input);

useGPU = ~isempty(opts.gpus);

if opts.cudnn, opts.cudnn = 'CuDNN'; else, opts.cudnn = 'NoCuDNN'; end

knn_idx = zeros([patchDims,opts.Nbrs,NI],'uint32');
if nargout > 1
  knn_D = cast(knn_idx,cid);
end

if useGPU
  clearMex();
  gpuDevice(opts.gpus(1));
  if ~isempty(opts.Wx)
    opts.Wx = cast(gpuArray(opts.Wx),cid);
  end
  if ~isempty(opts.Wy)
    opts.Wy = cast(gpuArray(opts.Wy),cid);
  end
end

for t =1:opts.batchSize:NI
  batchStart = t;
  batchEnd = min(t+opts.batchSize-1, NI);
  ind = batchStart:batchEnd;
  if useGPU
    input = gpuArray(im_net_input(:,:,:,ind));
  else 
    input = (im_net_input(:,:,:,ind));
  end
  
  Nc = size(input,3);
  if Nc > 1 && ~opts.multichannel_dist
    input = sum(input,3)/Nc;
  end
  
  if nargout > 1
    [idx,D] = misc.patchMatch(...
      nn_pad(input,opts.padSize,[],'padType',opts.padType),...
      'patchSize',opts.patchSize,'patchDist',opts.patchDist,...
      'Nbrs',opts.Nbrs,'searchwin',opts.searchwin,'stride',opts.stride,...
      'Wx',opts.Wx,'Wy',opts.Wy,'transform',opts.transform,'useSep',...
      opts.useSep,'sorted',opts.sorted,'cuDNN',opts.cudnn);
  else
    idx = misc.patchMatch(nn_pad(input,opts.padSize,[],...
      'padType',opts.padType),'patchSize',opts.patchSize,'patchDist', ...
      opts.patchDist,'Nbrs',opts.Nbrs,'searchwin',opts.searchwin, ...
      'stride',opts.stride,'Wx',opts.Wx,'Wy',opts.Wy,'transform', ...
      opts.transform,'useSep',opts.useSep,'sorted',opts.sorted, ...
      'cuDNN',opts.cudnn);
  end
  
  if useGPU
    knn_idx(:,:,:,ind) = gather(idx);
    if nargout > 1
      knn_D(:,:,:,ind) = gather(D);
    end
  else
    knn_idx(:,:,:,ind) = idx;
    if nargout > 1
      knn_D(:,:,:,ind) = D;
    end
  end
  
end


% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
%clear vl_tmove vl_imreadjpeg ;
disp('Clearing mex files');
clear mex;
clear vl_tmove vl_imreadjpeg;

