% Runs the cpu_processing measurement over grid.csv (ADR-0008). See README.md in this folder.
%
% Output (next to this file):
%   results/<host>_<yyyymmdd>_v<runVersion>/times.csv   one row per (configuration, seed, repetition)
%   results/<host>_<yyyymmdd>_v<runVersion>/env.txt     MATLAB version, machine, CPU, threads, settings
% The run can be stopped and started again: configurations already in times.csv are skipped.

%% Settings
seeds = 1:5;          % different transport blocks and noise
nWarm = 2;            % discarded warm-up repetitions per (configuration, seed)
nRep = 30;            % kept repetitions per (configuration, seed)
nRepLdpc = 5;         % of those, how many also time LDPC (calibration of k only)
ldpcMaxIter = 10;     % LDPC iterations: CloudRIC's FlexRAN CPU decoder always runs 10
ldpcTermination = 'max';   % 'max' = always ldpcMaxIter (like FlexRAN); 'early' = stop when parity checks pass
runVersion = 3;       % v1: early-termination LDPC; v2: ran on 10 threads (stale onCleanup); bump for a fresh folder
nGlobalWarm = 20;     % untimed runs of the first configuration before measuring (JIT / caches)

%% Paths and environment
here = fileparts(mfilename('fullpath'));
cfgs = readtable(fullfile(here, 'grid.csv'));
[~, host] = system('hostname'); host = strtrim(host);
outDir = fullfile(here, 'results', sprintf('%s_%s_v%d', host, datestr(now, 'yyyymmdd'), runVersion));
if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, 'times.csv');

% This is a script: an onCleanup left in the workspace by a previous run would fire when
% `cleanup` is reassigned and restore the old thread count (that is what broke v2). So the old
% object is cleared first, and one thread is set only after the new one exists.
clear cleanup
nThreads = maxNumCompThreads;
cleanup = onCleanup(@() maxNumCompThreads(nThreads));
maxNumCompThreads(1);
assert(maxNumCompThreads == 1, 'could not set MATLAB to one computational thread');
if ismac
    [~, cpu] = system('sysctl -n machdep.cpu.brand_string');
elseif isunix
    [~, cpu] = system('lscpu | grep "Model name"');
else
    [~, cpu] = system('wmic cpu get name');
end
fid = fopen(fullfile(outDir, 'env.txt'), 'w');
fprintf(fid, 'matlab: %s\ncomputer: %s\nhost: %s\ncpu: %s\nthreads: %d (was %d)\n', ...
    version, computer, host, strtrim(cpu), maxNumCompThreads, nThreads);
fprintf(fid, 'seeds: %s\nnWarm: %d\nnRep: %d\nnRepLdpc: %d\nldpcMaxIter: %d\nldpcTermination: %s\nnGlobalWarm: %d\nstarted: %s\n', ...
    mat2str(seeds), nWarm, nRep, nRepLdpc, ldpcMaxIter, ldpcTermination, nGlobalWarm, datestr(now));
fclose(fid);

%% Skip what is already measured
done = false(height(cfgs), numel(seeds));
if isfile(outFile)
    prev = readtable(outFile);
    for i = 1:height(cfgs)
        for s = 1:numel(seeds)
            done(i, s) = any(prev.mcs == cfgs.mcs(i) & prev.nprb == cfgs.nprb(i) & prev.seed == seeds(s));
        end
    end
end

%% Global warm-up (v1 showed the first configurations 2-4x slower): results discarded
for w = 1:nGlobalWarm
    measureCpuProcessing(cfgs(1 + mod(w - 1, height(cfgs)), :), 0, 0, 1, 1, ldpcMaxIter, ldpcTermination);
end

%% Run
t0 = tic; nDone = 0; nTodo = nnz(~done);
for i = 1:height(cfgs)
    for s = 1:numel(seeds)
        if done(i, s), continue; end
        T = measureCpuProcessing(cfgs(i, :), seeds(s), nWarm, nRep, nRepLdpc, ldpcMaxIter, ldpcTermination);
        writetable(T, outFile, 'WriteMode', 'append');   % header only when the file is new
        nDone = nDone + 1;
        eta = toc(t0) / nDone * (nTodo - nDone);
        fprintf('[%d/%d] MCS %2d nPRB %3d seed %d  pre-LDPC median %.3f ms  CRC %d  ETA %.1f min\n', ...
            nDone, nTodo, cfgs.mcs(i), cfgs.nprb(i), seeds(s), ...
            median(T.t_chest_ms + T.t_extract_ms + T.t_eq_ms + T.t_demod_ms + T.t_ratercv_ms), ...
            min(T.crc_ok), eta / 60);
    end
end
fid = fopen(fullfile(outDir, 'env.txt'), 'a'); fprintf(fid, 'finished: %s\n', datestr(now)); fclose(fid);
