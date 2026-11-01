function T = measureCpuProcessing(cfg, seed, nWarm, nRep, nRepLdpc, ldpcMaxIter, ldpcTermination)
% Times the cpu_processing blocks of one PUSCH grant (ADR-0008), plus LDPC as calibration anchor.
%
%   T = measureCpuProcessing(cfg, seed, nWarm, nRep, nRepLdpc, ldpcMaxIter, ldpcTermination)
%
% ldpcTermination: 'max' (always ldpcMaxIter iterations, like CloudRIC's FlexRAN CPU decoder,
% which always reports 10) or 'early'.
%
% cfg: one row of grid.csv (mcs, qm, r_x1024, nprb, tbs, base_graph, n_cb, cb_len, total_bits,
% g_bits, snr_db). Numerology of the simulator: 30 kHz SCS, 273-PRB carrier, 14 symbols,
% 1 DMRS symbol with 2 CDM groups without data (156 RE/PRB), 1 layer, 1 TX and 1 RX antenna.
% Channel: AWGN on the resource grid (OFDM modulation / demodulation are not part of
% cpu_processing, so they are skipped).
%
% Timed blocks (cpu_processing = 2 + 3 + 4 + 5 + 6):
%   2 t_chest_ms     nrChannelEstimate     DMRS channel estimation
%   3 t_extract_ms   nrExtractResources    PUSCH REs of the grant
%   4 t_eq_ms        nrEqualizeMMSE        MMSE equalization
%   5 t_demod_ms     nrPUSCHDecode         layer demapping + soft demodulation (LLR) + descrambling
%   6 t_ratercv_ms   nrRateRecoverLDPC     rate recovery
%   - t_ldpc_ms      nrLDPCDecode          NOT cpu_processing: only to calibrate k against CloudRIC
%                                          (fixed ldpcMaxIter iterations with ldpcTermination = 'max')
%
% Returns one row per repetition (nRep rows); t_ldpc_ms is NaN after the first nRepLdpc reps.
% Every repetition re-runs the whole chain on the same received grid, each block under its own
% tic/toc; the first nWarm repetitions are discarded (warm-up).

assert(maxNumCompThreads == 1, 'measure on one computational thread (maxNumCompThreads(1))');
rng(seed);
modNames = containers.Map({2, 4, 6, 8}, {'QPSK', '16QAM', '64QAM', '256QAM'});
modulation = modNames(cfg.qm);
R = cfg.r_x1024 / 1024;

%% Carrier and PUSCH
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30;
carrier.NSizeGrid = 273;
carrier.CyclicPrefix = 'Normal';
carrier.NCellID = 0;

pusch = nrPUSCHConfig;
pusch.PRBSet = 0:cfg.nprb - 1;
pusch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
pusch.MappingType = 'A';
pusch.NID = carrier.NCellID;
pusch.RNTI = 2;
pusch.TransformPrecoding = false;
pusch.NumLayers = 1;
pusch.NumAntennaPorts = 1;
pusch.TransmissionScheme = 'nonCodebook';
pusch.Modulation = modulation;
pusch.DMRS.DMRSConfigurationType = 1;
pusch.DMRS.DMRSTypeAPosition = 2;
pusch.DMRS.DMRSLength = 1;
pusch.DMRS.DMRSAdditionalPosition = 0;
pusch.DMRS.NumCDMGroupsWithoutData = 2;

%% Same transport block as the simulator (asserts the sizes of grid.csv)
[puschIndices, info] = nrPUSCHIndices(carrier, pusch);
assert(info.NREPerPRB == 156, 'RE per PRB = %d, expected 156', info.NREPerPRB);
assert(info.G == cfg.g_bits, 'G = %d, grid says %d', info.G, cfg.g_bits);
tbs = nrTBS(modulation, 1, cfg.nprb, info.NREPerPRB, R);
assert(tbs == cfg.tbs, 'TBS = %d, grid says %d (MCS %d, nPRB %d)', tbs, cfg.tbs, cfg.mcs, cfg.nprb);
sch = nrULSCHInfo(tbs, R);
assert(sch.BGN == cfg.base_graph && sch.C == cfg.n_cb && sch.K == cfg.cb_len, ...
    'segmentation (BG %d, C %d, K %d) differs from grid.csv', sch.BGN, sch.C, sch.K);

%% Transmitter: UL-SCH -> PUSCH symbols + DMRS on the grid
enc = nrULSCH;
enc.TargetCodeRate = R;
setTransportBlock(enc, randi([0 1], tbs, 1));
cw = enc(modulation, 1, info.G, 0);
txGrid = nrResourceGrid(carrier, 1);
txGrid(puschIndices) = nrPUSCH(carrier, pusch, cw);
dmrsIndices = nrPUSCHDMRSIndices(carrier, pusch);
dmrsSymbols = nrPUSCHDMRS(carrier, pusch);
txGrid(dmrsIndices) = dmrsSymbols;

%% AWGN on the grid (unit-power symbols)
nVar = 10^(-cfg.snr_db / 10);
rxGrid = txGrid + sqrt(nVar / 2) * complex(randn(size(txGrid)), randn(size(txGrid)));

%% Receiver: the timed chain
names = {'t_chest_ms', 't_extract_ms', 't_eq_ms', 't_demod_ms', 't_ratercv_ms', 't_ldpc_ms'};
times = nan(nWarm + nRep, numel(names));
iters = nan(nWarm + nRep, 1);
crcOK = nan(nWarm + nRep, 1);
for r = 1:(nWarm + nRep)
    t0 = tic; [hEst, nEst] = nrChannelEstimate(carrier, rxGrid, dmrsIndices, dmrsSymbols, 'AveragingWindow', [1 3]);
    times(r, 1) = toc(t0);
    t0 = tic; [puschRx, puschH] = nrExtractResources(puschIndices, rxGrid, hEst);
    times(r, 2) = toc(t0);
    t0 = tic; puschEq = nrEqualizeMMSE(puschRx, puschH, nEst);
    times(r, 3) = toc(t0);
    t0 = tic; llr = nrPUSCHDecode(carrier, pusch, puschEq, nEst);
    times(r, 4) = toc(t0);
    t0 = tic; rcv = nrRateRecoverLDPC(llr, tbs, R, 0, modulation, 1);
    times(r, 5) = toc(t0);
    if r <= nWarm + nRepLdpc
        t0 = tic; [dec, nIter] = nrLDPCDecode(rcv, sch.BGN, ldpcMaxIter, 'Algorithm', 'Normalized min-sum', ...
            'Termination', ldpcTermination);
        times(r, 6) = toc(t0);
        iters(r) = max(nIter);
        blk = nrCodeBlockDesegmentLDPC(dec, sch.BGN, tbs + sch.L);
        [~, crcErr] = nrCRCDecode(blk, sch.CRC);
        crcOK(r) = double(crcErr == 0);
    end
end

%% One row per kept repetition
keep = (nWarm + 1):(nWarm + nRep);
n = numel(keep);
T = table(repmat(cfg.mcs, n, 1), repmat(cfg.qm, n, 1), repmat(cfg.r_x1024, n, 1), repmat(cfg.nprb, n, 1), ...
    repmat(tbs, n, 1), repmat(sch.BGN, n, 1), repmat(sch.C, n, 1), repmat(sch.K, n, 1), ...
    repmat(cfg.total_bits, n, 1), repmat(info.G, n, 1), repmat(cfg.snr_db, n, 1), ...
    repmat(seed, n, 1), (1:n)', ...
    'VariableNames', {'mcs', 'qm', 'r_x1024', 'nprb', 'tbs', 'base_graph', 'n_cb', 'cb_len', ...
                      'total_bits', 'g_bits', 'snr_db', 'seed', 'rep'});
for k = 1:numel(names)
    T.(names{k}) = 1e3 * times(keep, k);
end
T.ldpc_iters = iters(keep);
T.crc_ok = crcOK(keep);
end
