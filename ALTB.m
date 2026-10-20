function simulate_ALB_AHSA_pubready_offload_fixed_with_baselines()
% simulate_ALB_AHSA_pubready_offload_fixed_with_baselines()
% Monte Carlo simulation of ATLB (adaptive TN-NTN load balancing) and AHSA,
% together with four baselines: TN-only, NTN-only, Threshold and Q-learning.
% Outputs per scheme: energy, delay, packet loss, NTN usage fraction and the
% load-balancing metric (Delta N), for several satellite counts (M) and numbers
% of devices (N), averaged over MC_runs independent runs.
%
% Model summary:
%   - Rician fading for TN and NTN: channel -> SINR -> detection probability Pd
%     -> success probability Ps = Pd*(1-Pcoll).
%   - Collision probability follows from the number of contending devices (pre-pass).
%   - NTN transmission energy is scaled by ntn_cost_alpha; mobility cost is normalized.
%   - Per-device delay is saturated at D_max.
%
% Threshold baseline:
%   - Fixed SINR thresholds select TN or NTN from the measured SINR_TN and SINR_NTN.
%   - TN threshold: 3 dB; NTN threshold: -3 dB.
%   - No battery awareness, no load distribution, no buffering.
%
% Q-learning (RL) baseline:
%   - Tabular Q-learning agent for network selection.
%   - State: discretized (SINR_TN, SINR_NTN, load_diff), 5x5x5 = 125 states.
%   - Actions: TN (1), NTN (2), no assignment (3, counted as lost).
%   - Reward: negative energy per message, plus a penalty for lost messages.
%   - Learning rate 0.1, discount factor 0.95, exploration rate 0.1.

%% ==================== USER PARAMETERS ====================
tic;  % start timer
clear; close all; clc;
rng_seed = 12345; rng(rng_seed);


Nsats_list_main  = [12, 100, 200, 300];   % main six-scheme comparison (four densities)
Nsats_list       = [12, 300];             % K-sweep, delta/Bmin/tau sweeps, Rayleigh/Markov
Nsats_list_extra = [100, 200];            % additional densities for the main comparison

ED_list    = [50, 100, 200, 500, 1000, 2000, 5000];
MC_runs    = 200;

% ALB params
tau_SINR = 0; B_min = 0.2; delta = 0.15; hybrid_link_prob = 0.5;
TN_capacity = 200; sat_capacity_per_sat = 5;

% Sensitivity sweeps for the ATLB parameters (one at a time)
delta_sweep    = [0.05, 0.15, 0.30];   % baseline 0.15
Bmin_sweep     = [0.10, 0.20, 0.35];   % baseline 0.20
tauSINR_sweep  = [-3,   0,    3   ];   % baseline 0 dB

% HCGS / PHY / RA
Nrep_set = [1,2,4,8];
D_max = 50; max_attempts = 8;
semi_persistent_threshold = 0.99;

% ==================== HARD PRE-CLEAN (Rayleigh files) ====================
% Prevent loading stale failed outputs from older runs
ray_files = { ...
    'results_rayleigh_static_TN_NTN.mat', ...
    'results_rayleigh_accuracy_checkpoint.mat'};

for f = 1:numel(ray_files)
    if isfile(ray_files{f})
        delete(ray_files{f});
        fprintf('[Pre-clean] Deleted %s\n', ray_files{f});
    end
end

% ==================== RICIAN FADING PARAMETERS ====================
% Terrestrial link: Rayleigh fading (K = 0 dB), urban environment
TN_SINR_mu = 10; TN_SINR_sigma = 6;
TN_pathloss_offset_dB = -160;              % ~160 dB at cell edge
K_factor_TN_dB = 0;                        % Rayleigh TN (K = 0 dB)
K_factor_TN = 10^(K_factor_TN_dB/10);      % = 1

% Non-Terrestrial: Satellite link, weaker LOS, K = 3 dB (~2 linear)
% 600 km altitude LEO, ~165 dB path loss
NTN_SINR_mu = 0; NTN_SINR_sigma = 2;
NTN_pathloss_offset_dB = -165;

K_factor_NTN_dB = 3;
K_factor_NTN_dB_sweep = [0, 3, 4, 7, 10];
K_factor_NTN = 10^(K_factor_NTN_dB/10);    % ~2 linear

% K-model control:
% 'elevation' = use map_elev_to_K_NTN(theta)
% 'scalar'    = use the fixed K_factor_NTN
K_mode = 'elevation';

% Detection model: SAME for both TN and NTN
% Pd = 1/(1 + exp(-slope*(SINR - threshold)))
detection_slope = 0.3;       % Sigmoid curvature (controls steepness)
detection_threshold = 3;     % SINR threshold (dB) where Pd ≈ 0.5

% Threshold baseline parameters
TN_threshold_dB = 3;         % Fixed TN SINR threshold for selection (dB)
NTN_threshold_dB = -3;       % Fixed NTN SINR threshold for selection (dB)

% Q-learning baseline parameters
RL_alpha = 0.1;              % Learning rate for Q-learning
RL_gamma = 0.95;             % Discount factor
RL_epsilon = 0.1;            % Exploration rate (ε-greedy)
RL_num_states = 125;         % 5x5x5 discretized state space (SINR_TN x SINR_NTN x load_diff)
RL_num_actions = 3;          % Action space: TN=1, NTN=2, no assignment=3
RL_train_episodes = 10000;
RL_eval_epsilon   = 0.01;
RL_eta_loss       = 2.0;

% RA parameters
L_preambles = 64;  % NPRACH preamble pool
capture_ratio_dB = 3; capture_ratio = 10^(capture_ratio_dB/10);
noise_floor = 1e-6;

% Energy/mobility
E_tx_TN  = 5e-4; E_tx_NTN = 1e-3;
E_idle_TN = 1e-5; E_idle_NTN = 2e-5;
E_rx = 3e-4;
k_mob = 5e-5;
mob_cap = 1e-2;

% AHSA adaptation
num_epochs = 12; beta_adapt = 0.6; r_star = 0.12;
p_min = 0.05; p_max = 0.9; Tmin_backoff = 1; alpha_backoff = 12;
frac_TN_only = 0.30; frac_NTN_only = 0.30; frac_dual = 0.40;
period_mean = 4; period_std = 1.5;
period_mean_rayleigh = 40;
period_std_rayleigh  = 15;

baseline_NED_ref = 1000;
arrival_scale = baseline_NED_ref / ED_list(end);
period_mean = max(1, period_mean * arrival_scale);
fprintf('period_mean=%.2f period_std=%.2f\n', period_mean, period_std);

% NTN cost enforcement
ntn_cost_alpha = 1.2;
ntn_prop_delay = 4;

% ==================== MARKOV MAPPING CONTROLS (EXPLICIT) ====================
mk_enable = true;
mk_use_sg_pd = false;            % true only if SG constants are explicitly validated

mk_lambda_mode = 'from_reporting_period';   % lambda = N / T_slots_eff (T from period_mean/period_std)

mk_T_OFF_mode = 'from_visibility_bernoulli'; % T_OFF = (1-pNTN)/pNTN
mk_T_OFF_fixed_slots = NaN;                  % used only if mk_T_OFF_mode='fixed'

mk_service_mode = 'per_device';   % 'per_device' or 'aggregate'

mk_connectivity_mode = 'tn_always_on';  % 'tn_always_on' or 'both_can_out'

mk_pi_tol = 1e-9;          % convergence tolerance
mk_pi_maxit = 100000;      % maximum iterations
mk_pi_require_convergence = true;

% SG constants, only used if mk_use_sg_pd = true
mk_sigma2 = NaN;
mk_alpha  = NaN;
mk_rhoPPP = NaN;
mk_Omega_TN = NaN;
mk_Omega_NTN = NaN;

% NOTE: mk_connectivity_mode='tn_always_on' assumes that NTN visibility is
% intermittent while TN is continuously available.

% Plotting
colorblind = true; export_figures = true;
z95 = 1.96;

% ===== Publication run control =====
run_publication_only = false;   % true: only one deterministic canonical run
canonical_K_NTN_dB = 3;        % nominal tag only

% === Sensitivity/validation plotting options ===
export_sensitivity_figs = true;   % set to false to skip the sensitivity/Rayleigh figures


markov_delay_TN   = [];
markov_delay_NTN  = [];
markov_loss_TN    = [];
markov_loss_NTN   = [];

% Keep immutable baseline ATLB parameters for one-at-a-time sweeps
tau_SINR_base = tau_SINR;
B_min_base    = B_min;
delta_base    = delta;

origAxesFS = get(groot,'defaultAxesFontSize');
set(groot,'defaultAxesFontSize', origAxesFS + 2);
try
    origLegendFS = get(groot,'defaultLegendFontSize');
    set(groot,'defaultLegendFontSize', origLegendFS + 2);
catch
end

%% ==================== PREPARE OUTPUTS ====================
nN = numel(Nsats_list);              % sensitivity/Rayleigh paths
nN_main = numel(Nsats_list_main);    % main-comparison paths
nE = numel(ED_list);

% ALB metrics
ALB_energy_mean = zeros(nN,nE); ALB_energy_ci = zeros(nN,nE);
ALB_delay_mean  = zeros(nN,nE); ALB_delay_ci  = zeros(nN,nE);
ALB_loss_mean   = zeros(nN,nE); ALB_loss_ci   = zeros(nN,nE);
ALB_emob_mean   = zeros(nN,nE); ALB_emob_ci   = zeros(nN,nE);
ALB_off_mean    = zeros(nN,nE); ALB_off_ci    = zeros(nN,nE);
ALB_lb_mean     = zeros(nN,nE); ALB_lb_ci     = zeros(nN,nE);

ALB_delay_std_mean = zeros(nN,nE); ALB_delay_std_ci = zeros(nN,nE);
ALB_delay_p95_mean = zeros(nN,nE); ALB_delay_p95_ci = zeros(nN,nE);
ALB_delay_p99_mean = zeros(nN,nE); ALB_delay_p99_ci = zeros(nN,nE);
ALB_delay_max_mean = zeros(nN,nE); ALB_delay_max_ci = zeros(nN,nE);

% AHSA metrics
AHSA_energy_mean = zeros(nN,nE); AHSA_energy_ci = zeros(nN,nE);
AHSA_delay_mean  = zeros(nN,nE); AHSA_delay_ci  = zeros(nN,nE);
AHSA_loss_mean   = zeros(nN,nE); AHSA_loss_ci   = zeros(nN,nE);
AHSA_emob_mean   = zeros(nN,nE); AHSA_emob_ci   = zeros(nN,nE);
AHSA_off_mean    = zeros(nN,nE); AHSA_off_ci    = zeros(nN,nE);
AHSA_lb_mean     = zeros(nN,nE); AHSA_lb_ci     = zeros(nN,nE);

AHSA_delay_std_mean = zeros(nN,nE); AHSA_delay_std_ci = zeros(nN,nE);
AHSA_delay_p95_mean = zeros(nN,nE); AHSA_delay_p95_ci = zeros(nN,nE);
AHSA_delay_p99_mean = zeros(nN,nE); AHSA_delay_p99_ci = zeros(nN,nE);
AHSA_delay_max_mean = zeros(nN,nE); AHSA_delay_max_ci = zeros(nN,nE);

% TN-only baseline metrics
TNonly_energy_mean = zeros(nN,nE); TNonly_energy_ci = zeros(nN,nE);
TNonly_delay_mean  = zeros(nN,nE); TNonly_delay_ci  = zeros(nN,nE);
TNonly_loss_mean   = zeros(nN,nE); TNonly_loss_ci   = zeros(nN,nE);
TNonly_emob_mean   = zeros(nN,nE); TNonly_emob_ci   = zeros(nN,nE);
TNonly_off_mean    = zeros(nN,nE); TNonly_off_ci    = zeros(nN,nE);
TNonly_lb_mean     = zeros(nN,nE); TNonly_lb_ci     = zeros(nN,nE);

% NTN-only baseline metrics
NTNonly_energy_mean = zeros(nN,nE); NTNonly_energy_ci = zeros(nN,nE);
NTNonly_delay_mean  = zeros(nN,nE); NTNonly_delay_ci  = zeros(nN,nE);
NTNonly_loss_mean   = zeros(nN,nE); NTNonly_loss_ci   = zeros(nN,nE);
NTNonly_emob_mean   = zeros(nN,nE); NTNonly_emob_ci   = zeros(nN,nE);
NTNonly_off_mean    = zeros(nN,nE); NTNonly_off_ci    = zeros(nN,nE);
NTNonly_lb_mean     = zeros(nN,nE); NTNonly_lb_ci     = zeros(nN,nE);

% Threshold baseline metrics
ThresholdBL_energy_mean = zeros(nN,nE); ThresholdBL_energy_ci = zeros(nN,nE);
ThresholdBL_delay_mean  = zeros(nN,nE); ThresholdBL_delay_ci  = zeros(nN,nE);
ThresholdBL_loss_mean   = zeros(nN,nE); ThresholdBL_loss_ci   = zeros(nN,nE);
ThresholdBL_emob_mean   = zeros(nN,nE); ThresholdBL_emob_ci   = zeros(nN,nE);
ThresholdBL_off_mean    = zeros(nN,nE); ThresholdBL_off_ci    = zeros(nN,nE);
ThresholdBL_lb_mean     = zeros(nN,nE); ThresholdBL_lb_ci     = zeros(nN,nE);

% Q-learning baseline metrics
RL_energy_mean = zeros(nN,nE); RL_energy_ci = zeros(nN,nE);
RL_delay_mean  = zeros(nN,nE); RL_delay_ci  = zeros(nN,nE);
RL_loss_mean   = zeros(nN,nE); RL_loss_ci   = zeros(nN,nE);
RL_emob_mean   = zeros(nN,nE); RL_emob_ci   = zeros(nN,nE);
RL_off_mean    = zeros(nN,nE); RL_off_ci    = zeros(nN,nE);
RL_lb_mean     = zeros(nN,nE); RL_lb_ci     = zeros(nN,nE);

% P-values (ALB vs AHSA)
p_energy = nan(nN, nE);
p_delay  = nan(nN, nE);
p_loss   = nan(nN, nE);

eff_energy = nan(nN, nE);
eff_delay  = nan(nN, nE);
eff_loss   = nan(nN, nE);

% Fixed algorithm color map
% ATLB: b, AHSA: r, TN-only: g, NTN-only: c, Threshold: k, Q-learning: m
clr_ATLB = 'b';
clr_AHSA = 'r';
clr_TN   = 'g';
clr_NTN  = 'c';
clr_TH   = 'k';
clr_QL   = 'm';

%% ==================== MAIN SIMULATION LOOPS ====================
fprintf('Running main simulation (MC_runs=%d)...\n', MC_runs);
results_plot = struct();   % initialize holder for plotting dataset

% sensitivity result containers
results_delta   = struct([]);
results_Bmin    = struct([]);
results_tauSINR = struct([]);

if ~run_publication_only
    for iK = 1:numel(K_factor_NTN_dB_sweep)
        K_factor_NTN_dB = K_factor_NTN_dB_sweep(iK);
        K_factor_NTN = 10^(K_factor_NTN_dB/10);
        K_offset_dB = K_factor_NTN_dB - canonical_K_NTN_dB;
        K_mode_loop = 'elevation';   % keep elevation model, but shifted by K_offset_dB

        for iN = 1:nN
            Nsats = Nsats_list(iN);
            p_ntn_cover = min(1, Nsats/300);
            cap_NTN = sat_capacity_per_sat * Nsats;

            ALB_energy_runs = zeros(MC_runs,nE);
            ALB_delay_runs  = zeros(MC_runs,nE);
            ALB_loss_runs   = zeros(MC_runs,nE);
            ALB_emob_runs   = zeros(MC_runs,nE);
            ALB_off_runs    = zeros(MC_runs,nE);
            ALB_lb_runs     = zeros(MC_runs,nE);

% Debug accumulators (ALB mechanism tracing)
            ALB_dbg_ntn_try_runs            = zeros(MC_runs,nE);
            ALB_dbg_ntn_accept_runs         = zeros(MC_runs,nE);
            ALB_dbg_ntn_reject_gate_runs    = zeros(MC_runs,nE);
            ALB_dbg_ntn_reject_batt_runs    = zeros(MC_runs,nE);
            ALB_dbg_fallback_tn_runs        = zeros(MC_runs,nE);
            ALB_dbg_buffer_entries_runs     = zeros(MC_runs,nE);
            ALB_dbg_buffer_success_tn_runs  = zeros(MC_runs,nE);
            ALB_dbg_buffer_success_ntn_runs = zeros(MC_runs,nE);
            ALB_dbg_buffer_drop_runs        = zeros(MC_runs,nE);
            ALB_dbg_tn_direct_runs          = zeros(MC_runs,nE);
            ALB_dbg_pcoll_ntn_runs          = zeros(MC_runs,nE); % from N_active_NTN pre-pass

            ALB_delay_std_runs  = zeros(MC_runs,nE);
            ALB_delay_p95_runs  = zeros(MC_runs,nE);
            ALB_delay_p99_runs  = zeros(MC_runs,nE);
            ALB_delay_max_runs  = zeros(MC_runs,nE);

            AHSA_energy_runs = zeros(MC_runs,nE);
            AHSA_delay_runs  = zeros(MC_runs,nE);
            AHSA_loss_runs   = zeros(MC_runs,nE);
            AHSA_emob_runs   = zeros(MC_runs,nE);
            AHSA_off_runs    = zeros(MC_runs,nE);
            AHSA_lb_runs     = zeros(MC_runs,nE);

            AHSA_delay_std_runs  = zeros(MC_runs,nE);
            AHSA_delay_p95_runs  = zeros(MC_runs,nE);
            AHSA_delay_p99_runs  = zeros(MC_runs,nE);
            AHSA_delay_max_runs  = zeros(MC_runs,nE);

            TNonly_energy_runs = zeros(MC_runs,nE);
            TNonly_delay_runs  = zeros(MC_runs,nE);
            TNonly_loss_runs   = zeros(MC_runs,nE);
            TNonly_emob_runs   = zeros(MC_runs,nE);
            TNonly_off_runs    = zeros(MC_runs,nE);
            TNonly_lb_runs     = zeros(MC_runs,nE);

            NTNonly_energy_runs = zeros(MC_runs,nE);
            NTNonly_delay_runs  = zeros(MC_runs,nE);
            NTNonly_loss_runs   = zeros(MC_runs,nE);
            NTNonly_emob_runs   = zeros(MC_runs,nE);
            NTNonly_off_runs    = zeros(MC_runs,nE);
            NTNonly_lb_runs     = zeros(MC_runs,nE);

% Initialize Threshold baseline runs
            ThresholdBL_energy_runs = zeros(MC_runs,nE);
            ThresholdBL_delay_runs  = zeros(MC_runs,nE);
            ThresholdBL_loss_runs   = zeros(MC_runs,nE);
            ThresholdBL_emob_runs   = zeros(MC_runs,nE);
            ThresholdBL_off_runs    = zeros(MC_runs,nE);
            ThresholdBL_lb_runs     = zeros(MC_runs,nE);

% Initialize RL baseline runs
            RL_energy_runs = zeros(MC_runs,nE);
            RL_delay_runs  = zeros(MC_runs,nE);
            RL_loss_runs   = zeros(MC_runs,nE);
            RL_emob_runs   = zeros(MC_runs,nE);
            RL_off_runs    = zeros(MC_runs,nE);
            RL_lb_runs     = zeros(MC_runs,nE);

            % ==================== RL: TRAIN ONCE PER Nsats (frozen during MC eval) ====================
            Q_table_RL = zeros(RL_num_states, RL_num_actions);
            eps_sched = linspace(0.30, 0.02, RL_train_episodes);

            for ep = 1:RL_train_episodes
                NED_tr = ED_list(randi(nE));
                caps_tr = mnrnd_vector(NED_tr, [frac_TN_only, frac_NTN_only, frac_dual]);

                [~,~,~,~,~,~,Q_table_RL] = simulate_RL_baseline_offload( ...
                    NED_tr, caps_tr, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                    TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                    NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                    K_factor_TN, K_factor_NTN, ...
                    detection_slope, detection_threshold, ...
                    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                    L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
                    RL_alpha, RL_gamma, eps_sched(ep), RL_num_states, RL_num_actions, ...
                    Q_table_RL, true, RL_eta_loss, K_mode_loop, K_offset_dB);
            end

            for run = 1:MC_runs
                NED_max = ED_list(end);
                caps_full = mnrnd_vector(NED_max, [frac_TN_only, frac_NTN_only, frac_dual]);
                T_hat_full = max(1, round(period_mean + period_std * randn(NED_max,1)));

                for iE = 1:nE
                    fprintf('Nsats %d/%d | Run %d/%d | ED %d/%d\r', iN,nN,run,MC_runs,iE,nE);

                    NED = ED_list(iE);
                    caps = caps_full(1:NED);
                    T_hat = T_hat_full(1:NED);

                    % ALB single-epoch
                    [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric, dbg_alb] = simulate_ALB_one_epoch_offload( ...
                        NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                        TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                        NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                        K_factor_TN, K_factor_NTN, ...
                        detection_slope, detection_threshold, ...
                        tau_SINR, B_min, delta, hybrid_link_prob, ...
                        Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                        E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                        L_preambles, ntn_cost_alpha, ntn_prop_delay, false, false, K_mode_loop, K_offset_dB);
                    ALB_energy_runs(run,iE) = totE;
                    ALB_delay_runs(run,iE)  = avgDelay;
                    ALB_loss_runs(run,iE)   = lossProb;
                    ALB_emob_runs(run,iE)   = totEmob / num_epochs;
                    ALB_off_runs(run,iE)    = off_frac;
                    ALB_lb_runs(run,iE)     = lb_metric;

                    ALB_delay_std_runs(run,iE) = stdDelay;
                    ALB_delay_p95_runs(run,iE) = p95Delay;
                    ALB_delay_p99_runs(run,iE) = p99Delay;
                    ALB_delay_max_runs(run,iE) = maxDelay;

% Persist ALB mechanism counters per (run,iE)
                    ALB_dbg_ntn_try_runs(run,iE)            = dbg_alb.ntn_try;
                    ALB_dbg_ntn_accept_runs(run,iE)         = dbg_alb.ntn_accept;
                    ALB_dbg_ntn_reject_gate_runs(run,iE)    = dbg_alb.ntn_reject_gate;
                    ALB_dbg_ntn_reject_batt_runs(run,iE)    = dbg_alb.ntn_reject_batt;
                    ALB_dbg_fallback_tn_runs(run,iE)        = dbg_alb.fallback_tn;
                    ALB_dbg_buffer_entries_runs(run,iE)     = dbg_alb.buffer_entries;
                    ALB_dbg_buffer_success_tn_runs(run,iE)  = dbg_alb.buffer_success_tn;
                    ALB_dbg_buffer_success_ntn_runs(run,iE) = dbg_alb.buffer_success_ntn;
                    ALB_dbg_buffer_drop_runs(run,iE)        = dbg_alb.buffer_drop;
                    ALB_dbg_tn_direct_runs(run,iE)          = dbg_alb.tn_direct;
                    
                    % collision proxy from pre-pass count in ALB function:
                    % Pcoll_NTN_fixed = 1 - (1 - 1/L_preambles)^(N_active_NTN_total - 1), if >1 else 0
                    if dbg_alb.ntn_accept > 1
                        ALB_dbg_pcoll_ntn_runs(run,iE) = 1 - (1 - 1/L_preambles)^(dbg_alb.ntn_accept - 1);
                    else
                        ALB_dbg_pcoll_ntn_runs(run,iE) = 0;
                    end

                    % AHSA multi-epoch
                    [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_AHSA_epochs_offload( ...
                        NED, caps, T_hat, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                        TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                        NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                        K_factor_TN, K_factor_NTN, ...
                        detection_slope, detection_threshold, ...
                        Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                        E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                        L_preambles, capture_ratio, noise_floor, ...
                        num_epochs, 0.5, 0.5, beta_adapt, r_star, p_min, p_max, Tmin_backoff, alpha_backoff, ...
                        ntn_cost_alpha, ntn_prop_delay, false, false, K_mode_loop, K_offset_dB);

                    AHSA_energy_runs(run,iE) = totE;
                    AHSA_delay_runs(run,iE)  = avgDelay;
                    AHSA_loss_runs(run,iE)   = lossProb;
                    AHSA_emob_runs(run,iE)   = totEmob / num_epochs;
                    AHSA_off_runs(run,iE)    = off_frac;
                    AHSA_lb_runs(run,iE)     = lb_metric;

                    AHSA_delay_std_runs(run,iE) = stdDelay;
                    AHSA_delay_p95_runs(run,iE) = p95Delay;
                    AHSA_delay_p99_runs(run,iE) = p99Delay;
                    AHSA_delay_max_runs(run,iE) = maxDelay;

                    % TN-only baseline
                    [totE, avgDelay, ~, ~, ~, ~, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                        NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                        TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                        NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                        K_factor_TN, K_factor_NTN, ...
                        detection_slope, detection_threshold, ...
                        tau_SINR, B_min, delta, hybrid_link_prob, ...
                        Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                        E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                        L_preambles, ntn_cost_alpha, ntn_prop_delay, true, false, K_mode_loop, K_offset_dB);
                    TNonly_energy_runs(run,iE) = totE;
                    TNonly_delay_runs(run,iE)  = avgDelay;
                    TNonly_loss_runs(run,iE)   = lossProb;
                    TNonly_emob_runs(run,iE)   = totEmob;
                    TNonly_off_runs(run,iE)    = off_frac;
                    TNonly_lb_runs(run,iE)     = lb_metric;

                    % NTN-only baseline
                    [totE, avgDelay, ~, ~, ~, ~, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                        NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                        TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                        NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                        K_factor_TN, K_factor_NTN, ...
                        detection_slope, detection_threshold, ...
                        tau_SINR, B_min, delta, hybrid_link_prob, ...
                        Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                        E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                        L_preambles, ntn_cost_alpha, ntn_prop_delay, false, true, K_mode_loop, K_offset_dB);
                    NTNonly_energy_runs(run,iE) = totE;
                    NTNonly_delay_runs(run,iE)  = avgDelay;
                    NTNonly_loss_runs(run,iE)   = lossProb;
                    NTNonly_emob_runs(run,iE)   = totEmob;
                    NTNonly_off_runs(run,iE)    = off_frac;
                    NTNonly_lb_runs(run,iE)     = lb_metric;

                    % Threshold baseline
                    [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_threshold_baseline_offload( ...
                        NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                        TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                        NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                        K_factor_TN, K_factor_NTN, ...
                        detection_slope, detection_threshold, ...
                        TN_threshold_dB, NTN_threshold_dB, ...
                        Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                        E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                        L_preambles, ntn_cost_alpha, ntn_prop_delay, K_mode_loop, K_offset_dB);

                    ThresholdBL_energy_runs(run,iE) = totE;
                    ThresholdBL_delay_runs(run,iE)  = avgDelay;
                    ThresholdBL_loss_runs(run,iE)   = lossProb;
                    ThresholdBL_emob_runs(run,iE)   = totEmob;
                    ThresholdBL_off_runs(run,iE)    = off_frac;
                    ThresholdBL_lb_runs(run,iE)     = lb_metric;

                    % RL baseline (EVALUATION with frozen Q-table)
                    [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric, Q_table_RL] = simulate_RL_baseline_offload( ...
                        NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                        TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                        NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                        K_factor_TN, K_factor_NTN, ...
                        detection_slope, detection_threshold, ...
                        Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                        E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                        L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
                        RL_alpha, RL_gamma, RL_eval_epsilon, RL_num_states, RL_num_actions, ...
                        Q_table_RL, false, RL_eta_loss, K_mode_loop, K_offset_dB);

                    RL_energy_runs(run,iE) = totE;
                    RL_delay_runs(run,iE)  = avgDelay;
                    RL_loss_runs(run,iE)   = lossProb;
                    RL_emob_runs(run,iE)   = totEmob;
                    RL_off_runs(run,iE)    = off_frac;
                    RL_lb_runs(run,iE)     = lb_metric;
                end

                fprintf('\n');
            end
% ==================== Mechanism log at the failing point (M=300, N=5000) ====================
            if Nsats == 300
                idxE_fail = find(ED_list == 5000, 1, 'first');
                if ~isempty(idxE_fail)
                    ntn_try_mean        = mean(ALB_dbg_ntn_try_runs(:,idxE_fail));
                    ntn_accept_mean     = mean(ALB_dbg_ntn_accept_runs(:,idxE_fail));
                    rej_gate_mean       = mean(ALB_dbg_ntn_reject_gate_runs(:,idxE_fail));
                    rej_batt_mean       = mean(ALB_dbg_ntn_reject_batt_runs(:,idxE_fail));
                    fallback_tn_mean    = mean(ALB_dbg_fallback_tn_runs(:,idxE_fail));
                    buffer_entries_mean = mean(ALB_dbg_buffer_entries_runs(:,idxE_fail));
                    buffer_drop_mean    = mean(ALB_dbg_buffer_drop_runs(:,idxE_fail));
                    pcoll_ntn_mean      = mean(ALB_dbg_pcoll_ntn_runs(:,idxE_fail));
            
                    diversion_mass = rej_gate_mean + fallback_tn_mean + buffer_drop_mean;
            
                    fprintf('\n[STEP4-MECH] M=300 N=5000 (ALB, averaged over %d MC runs)\n', MC_runs);
                    fprintf('  ntn_try=%.2f, ntn_accept=%.2f, reject_gate=%.2f, reject_batt=%.2f\n', ...
                        ntn_try_mean, ntn_accept_mean, rej_gate_mean, rej_batt_mean);
                    fprintf('  fallback_tn=%.2f, buffer_entries=%.2f, buffer_drop=%.2f\n', ...
                        fallback_tn_mean, buffer_entries_mean, buffer_drop_mean);
                    fprintf('  diversion_mass(reject_gate+fallback_tn+buffer_drop)=%.2f\n', diversion_mass);
                    fprintf('  NTN_collision_proxy_mean=%.4f\n', pcoll_ntn_mean);
            
                    if (diversion_mass > ntn_accept_mean) && (pcoll_ntn_mean < 0.2)
                        fprintf('  => STEP4 verdict: jump is diversion/gating-driven, not pure RA-collision-driven.\n');
                    else
                        fprintf('  => STEP4 verdict: inconclusive.\n');
                    end
                end
            end

            [ALB_energy_mean(iN,:), ALB_energy_ci(iN,:)] = mean_ci(ALB_energy_runs, z95);
            [ALB_delay_mean(iN,:),  ALB_delay_ci(iN,:)]  = mean_ci(ALB_delay_runs, z95);
            [ALB_loss_mean(iN,:),   ALB_loss_ci(iN,:)]   = mean_ci(ALB_loss_runs, z95);
            [ALB_emob_mean(iN,:),   ALB_emob_ci(iN,:)]   = mean_ci(ALB_emob_runs, z95);
            [ALB_off_mean(iN,:),    ALB_off_ci(iN,:)]    = mean_ci(ALB_off_runs, z95);
            [ALB_lb_mean(iN,:),     ALB_lb_ci(iN,:)]     = mean_ci(ALB_lb_runs, z95);

            [ALB_delay_std_mean(iN,:), ALB_delay_std_ci(iN,:)] = mean_ci(ALB_delay_std_runs, z95);
            [ALB_delay_p95_mean(iN,:), ALB_delay_p95_ci(iN,:)] = mean_ci(ALB_delay_p95_runs, z95);
            [ALB_delay_p99_mean(iN,:), ALB_delay_p99_ci(iN,:)] = mean_ci(ALB_delay_p99_runs, z95);
            [ALB_delay_max_mean(iN,:), ALB_delay_max_ci(iN,:)] = mean_ci(ALB_delay_max_runs, z95);

            [AHSA_energy_mean(iN,:), AHSA_energy_ci(iN,:)] = mean_ci(AHSA_energy_runs, z95);
            [AHSA_delay_mean(iN,:),  AHSA_delay_ci(iN,:)]  = mean_ci(AHSA_delay_runs, z95);
            [AHSA_loss_mean(iN,:),   AHSA_loss_ci(iN,:)]   = mean_ci(AHSA_loss_runs, z95);
            [AHSA_emob_mean(iN,:),   AHSA_emob_ci(iN,:)]   = mean_ci(AHSA_emob_runs, z95);
            [AHSA_off_mean(iN,:),    AHSA_off_ci(iN,:)]    = mean_ci(AHSA_off_runs, z95);
            [AHSA_lb_mean(iN,:),     AHSA_lb_ci(iN,:)]     = mean_ci(AHSA_lb_runs, z95);

            [AHSA_delay_std_mean(iN,:), AHSA_delay_std_ci(iN,:)] = mean_ci(AHSA_delay_std_runs, z95);
            [AHSA_delay_p95_mean(iN,:), AHSA_delay_p95_ci(iN,:)] = mean_ci(AHSA_delay_p95_runs, z95);
            [AHSA_delay_p99_mean(iN,:), AHSA_delay_p99_ci(iN,:)] = mean_ci(AHSA_delay_p99_runs, z95);
            [AHSA_delay_max_mean(iN,:), AHSA_delay_max_ci(iN,:)] = mean_ci(AHSA_delay_max_runs, z95);

            [TNonly_energy_mean(iN,:), TNonly_energy_ci(iN,:)] = mean_ci(TNonly_energy_runs, z95);
            [TNonly_delay_mean(iN,:),  TNonly_delay_ci(iN,:)]  = mean_ci(TNonly_delay_runs, z95);
            [TNonly_loss_mean(iN,:),   TNonly_loss_ci(iN,:)]   = mean_ci(TNonly_loss_runs, z95);
            [TNonly_emob_mean(iN,:),   TNonly_emob_ci(iN,:)]   = mean_ci(TNonly_emob_runs, z95);
            [TNonly_off_mean(iN,:),    TNonly_off_ci(iN,:)]    = mean_ci(TNonly_off_runs, z95);
            [TNonly_lb_mean(iN,:),     TNonly_lb_ci(iN,:)]     = mean_ci(TNonly_lb_runs, z95);

            [NTNonly_energy_mean(iN,:), NTNonly_energy_ci(iN,:)] = mean_ci(NTNonly_energy_runs, z95);
            [NTNonly_delay_mean(iN,:),  NTNonly_delay_ci(iN,:)]  = mean_ci(NTNonly_delay_runs, z95);
            [NTNonly_loss_mean(iN,:),   NTNonly_loss_ci(iN,:)]   = mean_ci(NTNonly_loss_runs, z95);
            [NTNonly_emob_mean(iN,:),   NTNonly_emob_ci(iN,:)]   = mean_ci(NTNonly_emob_runs, z95);
            [NTNonly_off_mean(iN,:),    NTNonly_off_ci(iN,:)]    = mean_ci(NTNonly_off_runs, z95);
            [NTNonly_lb_mean(iN,:),     NTNonly_lb_ci(iN,:)]     = mean_ci(NTNonly_lb_runs, z95);

% Compute statistics for Threshold baseline
            [ThresholdBL_energy_mean(iN,:), ThresholdBL_energy_ci(iN,:)] = mean_ci(ThresholdBL_energy_runs, z95);
            [ThresholdBL_delay_mean(iN,:),  ThresholdBL_delay_ci(iN,:)]  = mean_ci(ThresholdBL_delay_runs, z95);
            [ThresholdBL_loss_mean(iN,:),   ThresholdBL_loss_ci(iN,:)]   = mean_ci(ThresholdBL_loss_runs, z95);
            [ThresholdBL_emob_mean(iN,:),   ThresholdBL_emob_ci(iN,:)]   = mean_ci(ThresholdBL_emob_runs, z95);
            [ThresholdBL_off_mean(iN,:),    ThresholdBL_off_ci(iN,:)]    = mean_ci(ThresholdBL_off_runs, z95);
            [ThresholdBL_lb_mean(iN,:),     ThresholdBL_lb_ci(iN,:)]     = mean_ci(ThresholdBL_lb_runs, z95);

% Compute statistics for RL baseline
            [RL_energy_mean(iN,:), RL_energy_ci(iN,:)] = mean_ci(RL_energy_runs, z95);
            [RL_delay_mean(iN,:),  RL_delay_ci(iN,:)]  = mean_ci(RL_delay_runs, z95);
            [RL_loss_mean(iN,:),   RL_loss_ci(iN,:)]   = mean_ci(RL_loss_runs, z95);
            [RL_emob_mean(iN,:),   RL_emob_ci(iN,:)]   = mean_ci(RL_emob_runs, z95);
            [RL_off_mean(iN,:),    RL_off_ci(iN,:)]    = mean_ci(RL_off_runs, z95);
            [RL_lb_mean(iN,:),     RL_lb_ci(iN,:)]     = mean_ci(RL_lb_runs, z95);

            % Statistical significance testing (ALB vs AHSA)
            for iE = 1:nE
                try
                    [~, p_e, ~, eff_e, ~] = statistical_test(ALB_energy_runs(:,iE), AHSA_energy_runs(:,iE), 0.05);
                    p_energy(iN,iE)   = p_e;
                    eff_energy(iN,iE) = eff_e;
                catch
                    p_energy(iN,iE) = NaN;
                    eff_energy(iN,iE) = NaN;
                end
                try
                    [~, p_d, ~, eff_d, ~] = statistical_test(ALB_delay_runs(:,iE), AHSA_delay_runs(:,iE), 0.05);
                    p_delay(iN,iE)   = p_d;
                    eff_delay(iN,iE) = eff_d;
                catch
                    p_delay(iN,iE) = NaN;
                    eff_delay(iN,iE) = NaN;
                end
                try
                    [~, p_l, ~, eff_l, ~] = statistical_test(ALB_loss_runs(:,iE), AHSA_loss_runs(:,iE), 0.05);
                    p_loss(iN,iE)   = p_l;
                    eff_loss(iN,iE) = eff_l;
                catch
                    p_loss(iN,iE) = NaN;
                    eff_loss(iN,iE) = NaN;
                end
            end

            fprintf('  Completed M=%d\n', Nsats);
        end
        results_sensitivity(iK) = build_results_snapshot( ...
            ALB_energy_mean, ALB_delay_mean, ALB_loss_mean, ALB_off_mean, ...
            AHSA_energy_mean, AHSA_delay_mean, AHSA_loss_mean, AHSA_off_mean, ...
            TNonly_energy_mean, TNonly_delay_mean, TNonly_loss_mean, ...
            NTNonly_energy_mean, NTNonly_delay_mean, NTNonly_loss_mean, ...
            ThresholdBL_energy_mean, ThresholdBL_delay_mean, ThresholdBL_loss_mean, ...
            RL_energy_mean, RL_delay_mean, RL_loss_mean, ...
            Nsats_list, ED_list, K_factor_TN_dB, K_factor_NTN_dB);

        if abs(K_factor_NTN_dB - canonical_K_NTN_dB) < 1e-12
            results_plot = results_sensitivity(iK);   % keep canonical plotting set
        end
    end   % closes for iK
end   % closes if ~run_publication_only

%% ==================== ATLB PARAMETER SENSITIVITY (ONE-AT-A-TIME) ====================
% Runs ATLB only; all other algorithms are unchanged.

for iP = 1:numel(delta_sweep)
    tau_SINR = tau_SINR_base;
    B_min    = B_min_base;
    delta    = delta_sweep(iP);

    ALB_delay_mean_s = zeros(nN,nE); ALB_delay_ci_s = zeros(nN,nE);
    ALB_off_mean_s   = zeros(nN,nE); ALB_off_ci_s   = zeros(nN,nE);

    for iN = 1:nN
        Nsats = Nsats_list(iN);
        p_ntn_cover = min(1, Nsats/300);
        cap_NTN = sat_capacity_per_sat * Nsats;

        ALB_delay_runs_s = zeros(MC_runs,nE);
        ALB_off_runs_s   = zeros(MC_runs,nE);

        for run = 1:MC_runs
            NED_max = ED_list(end);
            caps_full = mnrnd_vector(NED_max, [frac_TN_only, frac_NTN_only, frac_dual]);

            for iE = 1:nE
                NED = ED_list(iE);
                caps = caps_full(1:NED);
                is_failpoint = (Nsats == 300) && (NED == 5000);

                [~, avgDelay, ~, ~, ~, ~, ~, ~, off_frac, ~, ~] = simulate_ALB_one_epoch_offload( ...
                    NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                    TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                    NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                    K_factor_TN, K_factor_NTN, ...
                    detection_slope, detection_threshold, ...
                    tau_SINR, B_min, delta, hybrid_link_prob, ...
                    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                    L_preambles, ntn_cost_alpha, ntn_prop_delay, false, false, K_mode, 0);

                ALB_delay_runs_s(run,iE) = avgDelay;
                ALB_off_runs_s(run,iE)   = off_frac;
            end
        end

        [ALB_delay_mean_s(iN,:), ALB_delay_ci_s(iN,:)] = mean_ci(ALB_delay_runs_s, z95);
        [ALB_off_mean_s(iN,:),   ALB_off_ci_s(iN,:)]   = mean_ci(ALB_off_runs_s, z95);
    end

    results_delta(iP).ALB_delay_mean = ALB_delay_mean_s;
    results_delta(iP).ALB_delay_ci   = ALB_delay_ci_s;
    results_delta(iP).ALB_off_mean   = ALB_off_mean_s;
    results_delta(iP).ALB_off_ci     = ALB_off_ci_s;
    results_delta(iP).param_value    = delta_sweep(iP);
end

for iP = 1:numel(Bmin_sweep)
    tau_SINR = tau_SINR_base;
    B_min    = Bmin_sweep(iP);
    delta    = delta_base;

    ALB_delay_mean_s = zeros(nN,nE); ALB_delay_ci_s = zeros(nN,nE);
    ALB_off_mean_s   = zeros(nN,nE); ALB_off_ci_s   = zeros(nN,nE);

    for iN = 1:nN
        Nsats = Nsats_list(iN);
        p_ntn_cover = min(1, Nsats/300);
        cap_NTN = sat_capacity_per_sat * Nsats;

        ALB_delay_runs_s = zeros(MC_runs,nE);
        ALB_off_runs_s   = zeros(MC_runs,nE);

        for run = 1:MC_runs
            NED_max = ED_list(end);
            caps_full = mnrnd_vector(NED_max, [frac_TN_only, frac_NTN_only, frac_dual]);

            for iE = 1:nE
                NED = ED_list(iE);
                caps = caps_full(1:NED);
                is_failpoint = (Nsats == 300) && (NED == 5000);

                [~, avgDelay, ~, ~, ~, ~, ~, ~, off_frac, ~, ~] = simulate_ALB_one_epoch_offload( ...
                    NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                    TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                    NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                    K_factor_TN, K_factor_NTN, ...
                    detection_slope, detection_threshold, ...
                    tau_SINR, B_min, delta, hybrid_link_prob, ...
                    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                    L_preambles, ntn_cost_alpha, ntn_prop_delay, false, false, K_mode, 0);

                ALB_delay_runs_s(run,iE) = avgDelay;
                ALB_off_runs_s(run,iE)   = off_frac;
            end
        end

        [ALB_delay_mean_s(iN,:), ALB_delay_ci_s(iN,:)] = mean_ci(ALB_delay_runs_s, z95);
        [ALB_off_mean_s(iN,:),   ALB_off_ci_s(iN,:)]   = mean_ci(ALB_off_runs_s, z95);
    end

    results_Bmin(iP).ALB_delay_mean = ALB_delay_mean_s;
    results_Bmin(iP).ALB_delay_ci   = ALB_delay_ci_s;
    results_Bmin(iP).ALB_off_mean   = ALB_off_mean_s;
    results_Bmin(iP).ALB_off_ci     = ALB_off_ci_s;
    results_Bmin(iP).param_value    = Bmin_sweep(iP);
end

for iP = 1:numel(tauSINR_sweep)
    tau_SINR = tauSINR_sweep(iP);
    B_min    = B_min_base;
    delta    = delta_base;

    ALB_delay_mean_s = zeros(nN,nE); ALB_delay_ci_s = zeros(nN,nE);
    ALB_off_mean_s   = zeros(nN,nE); ALB_off_ci_s   = zeros(nN,nE);

    for iN = 1:nN
        Nsats = Nsats_list(iN);
        p_ntn_cover = min(1, Nsats/300);
        cap_NTN = sat_capacity_per_sat * Nsats;

        ALB_delay_runs_s = zeros(MC_runs,nE);
        ALB_off_runs_s   = zeros(MC_runs,nE);

        for run = 1:MC_runs
            NED_max = ED_list(end);
            caps_full = mnrnd_vector(NED_max, [frac_TN_only, frac_NTN_only, frac_dual]);

            for iE = 1:nE
                NED = ED_list(iE);
                caps = caps_full(1:NED);
                is_failpoint = (Nsats == 300) && (NED == 5000);

                [~, avgDelay, ~, ~, ~, ~, ~, ~, off_frac, ~, ~] = simulate_ALB_one_epoch_offload( ...
                    NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                    TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                    NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                    K_factor_TN, K_factor_NTN, ...
                    detection_slope, detection_threshold, ...
                    tau_SINR, B_min, delta, hybrid_link_prob, ...
                    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                    L_preambles, ntn_cost_alpha, ntn_prop_delay, false, false, K_mode, 0);

                ALB_delay_runs_s(run,iE) = avgDelay;
                ALB_off_runs_s(run,iE)   = off_frac;
            end
        end

        [ALB_delay_mean_s(iN,:), ALB_delay_ci_s(iN,:)] = mean_ci(ALB_delay_runs_s, z95);
        [ALB_off_mean_s(iN,:),   ALB_off_ci_s(iN,:)]   = mean_ci(ALB_off_runs_s, z95);
    end

    results_tauSINR(iP).ALB_delay_mean = ALB_delay_mean_s;
    results_tauSINR(iP).ALB_delay_ci   = ALB_delay_ci_s;
    results_tauSINR(iP).ALB_off_mean   = ALB_off_mean_s;
    results_tauSINR(iP).ALB_off_ci     = ALB_off_ci_s;
    results_tauSINR(iP).param_value    = tauSINR_sweep(iP);
end

% restore baseline values after sweeps
tau_SINR = tau_SINR_base;
B_min    = B_min_base;
delta    = delta_base;

save('results_sensitivity_delta.mat','results_delta','Nsats_list','ED_list');
save('results_sensitivity_Bmin.mat','results_Bmin','Nsats_list','ED_list');
save('results_sensitivity_tauSINR.mat','results_tauSINR','Nsats_list','ED_list');

%% ==================== EXPORT MEAN ARRAYS ====================
% Optional: save full workspace (includes all *_mean, *_ci, runs, p-values, params)

if exist('results_sensitivity','var')
    save('results_sensitivity_KNTN.mat','results_sensitivity','Nsats_list','ED_list');
end

% ==================== SINGLE CANONICAL PUBLICATION RUN ====================
rng(rng_seed);  % deterministic publication outputs, independent of prior sweeps

K_factor_NTN_dB = canonical_K_NTN_dB;
K_factor_NTN = 10^(K_factor_NTN_dB/10);
K_offset_dB = 0;
K_mode_loop = 'elevation';   % publication model

% Reinitialize output arrays (single authoritative run)
ALB_energy_mean = zeros(nN,nE); ALB_energy_ci = zeros(nN,nE);
ALB_delay_mean  = zeros(nN,nE); ALB_delay_ci  = zeros(nN,nE);
ALB_loss_mean   = zeros(nN,nE); ALB_loss_ci   = zeros(nN,nE);
ALB_emob_mean   = zeros(nN,nE); ALB_emob_ci   = zeros(nN,nE);
ALB_off_mean    = zeros(nN,nE); ALB_off_ci    = zeros(nN,nE);
ALB_lb_mean     = zeros(nN,nE); ALB_lb_ci     = zeros(nN,nE);

AHSA_energy_mean = zeros(nN,nE); AHSA_energy_ci = zeros(nN,nE);
AHSA_delay_mean  = zeros(nN,nE); AHSA_delay_ci  = zeros(nN,nE);
AHSA_loss_mean   = zeros(nN,nE); AHSA_loss_ci   = zeros(nN,nE);
AHSA_emob_mean   = zeros(nN,nE); AHSA_emob_ci   = zeros(nN,nE);
AHSA_off_mean    = zeros(nN,nE); AHSA_off_ci    = zeros(nN,nE);
AHSA_lb_mean     = zeros(nN,nE); AHSA_lb_ci     = zeros(nN,nE);

TNonly_energy_mean = zeros(nN,nE); TNonly_energy_ci = zeros(nN,nE);
TNonly_delay_mean  = zeros(nN,nE); TNonly_delay_ci  = zeros(nN,nE);
TNonly_loss_mean   = zeros(nN,nE); TNonly_loss_ci   = zeros(nN,nE);
TNonly_emob_mean   = zeros(nN,nE); TNonly_emob_ci   = zeros(nN,nE);
TNonly_off_mean    = zeros(nN,nE); TNonly_off_ci    = zeros(nN,nE);
TNonly_lb_mean     = zeros(nN,nE); TNonly_lb_ci     = zeros(nN,nE);

NTNonly_energy_mean = zeros(nN,nE); NTNonly_energy_ci = zeros(nN,nE);
NTNonly_delay_mean  = zeros(nN,nE); NTNonly_delay_ci  = zeros(nN,nE);
NTNonly_loss_mean   = zeros(nN,nE); NTNonly_loss_ci   = zeros(nN,nE);
NTNonly_emob_mean   = zeros(nN,nE); NTNonly_emob_ci   = zeros(nN,nE);
NTNonly_off_mean    = zeros(nN,nE); NTNonly_off_ci    = zeros(nN,nE);
NTNonly_lb_mean     = zeros(nN,nE); NTNonly_lb_ci     = zeros(nN,nE);

ThresholdBL_energy_mean = zeros(nN,nE); ThresholdBL_energy_ci = zeros(nN,nE);
ThresholdBL_delay_mean  = zeros(nN,nE); ThresholdBL_delay_ci  = zeros(nN,nE);
ThresholdBL_loss_mean   = zeros(nN,nE); ThresholdBL_loss_ci   = zeros(nN,nE);
ThresholdBL_emob_mean   = zeros(nN,nE); ThresholdBL_emob_ci   = zeros(nN,nE);
ThresholdBL_off_mean    = zeros(nN,nE); ThresholdBL_off_ci    = zeros(nN,nE);
ThresholdBL_lb_mean     = zeros(nN,nE); ThresholdBL_lb_ci     = zeros(nN,nE);

RL_energy_mean = zeros(nN,nE); RL_energy_ci = zeros(nN,nE);
RL_delay_mean  = zeros(nN,nE); RL_delay_ci  = zeros(nN,nE);
RL_loss_mean   = zeros(nN,nE); RL_loss_ci   = zeros(nN,nE);
RL_emob_mean   = zeros(nN,nE); RL_emob_ci   = zeros(nN,nE);
RL_off_mean    = zeros(nN,nE); RL_off_ci    = zeros(nN,nE);
RL_lb_mean     = zeros(nN,nE); RL_lb_ci     = zeros(nN,nE);

for iN = 1:nN
    Nsats = Nsats_list(iN);
    p_ntn_cover = min(1, Nsats/300);
    cap_NTN = sat_capacity_per_sat * Nsats;

    ALB_energy_runs = zeros(MC_runs,nE); ALB_delay_runs  = zeros(MC_runs,nE); ALB_loss_runs   = zeros(MC_runs,nE);
    ALB_emob_runs   = zeros(MC_runs,nE); ALB_off_runs    = zeros(MC_runs,nE); ALB_lb_runs     = zeros(MC_runs,nE);
    ALB_delay_std_runs = zeros(MC_runs,nE); ALB_delay_p95_runs = zeros(MC_runs,nE);
    ALB_delay_p99_runs = zeros(MC_runs,nE); ALB_delay_max_runs = zeros(MC_runs,nE);

    AHSA_energy_runs = zeros(MC_runs,nE); AHSA_delay_runs  = zeros(MC_runs,nE); AHSA_loss_runs   = zeros(MC_runs,nE);
    AHSA_emob_runs   = zeros(MC_runs,nE); AHSA_off_runs    = zeros(MC_runs,nE); AHSA_lb_runs     = zeros(MC_runs,nE);
    AHSA_delay_std_runs = zeros(MC_runs,nE); AHSA_delay_p95_runs = zeros(MC_runs,nE);
    AHSA_delay_p99_runs = zeros(MC_runs,nE); AHSA_delay_max_runs = zeros(MC_runs,nE);

    TNonly_energy_runs = zeros(MC_runs,nE); TNonly_delay_runs = zeros(MC_runs,nE); TNonly_loss_runs = zeros(MC_runs,nE);
    TNonly_emob_runs   = zeros(MC_runs,nE); TNonly_off_runs   = zeros(MC_runs,nE); TNonly_lb_runs   = zeros(MC_runs,nE);

    NTNonly_energy_runs = zeros(MC_runs,nE); NTNonly_delay_runs = zeros(MC_runs,nE); NTNonly_loss_runs = zeros(MC_runs,nE);
    NTNonly_emob_runs   = zeros(MC_runs,nE); NTNonly_off_runs   = zeros(MC_runs,nE); NTNonly_lb_runs   = zeros(MC_runs,nE);

    ThresholdBL_energy_runs = zeros(MC_runs,nE); ThresholdBL_delay_runs = zeros(MC_runs,nE); ThresholdBL_loss_runs = zeros(MC_runs,nE);
    ThresholdBL_emob_runs   = zeros(MC_runs,nE); ThresholdBL_off_runs   = zeros(MC_runs,nE); ThresholdBL_lb_runs   = zeros(MC_runs,nE);

    RL_energy_runs = zeros(MC_runs,nE); RL_delay_runs = zeros(MC_runs,nE); RL_loss_runs = zeros(MC_runs,nE);
    RL_emob_runs   = zeros(MC_runs,nE); RL_off_runs   = zeros(MC_runs,nE); RL_lb_runs   = zeros(MC_runs,nE);

    Q_table_RL = zeros(RL_num_states, RL_num_actions);
    eps_sched = linspace(0.30, 0.02, RL_train_episodes);

    for ep = 1:RL_train_episodes
        NED_tr = ED_list(randi(nE));
        caps_tr = mnrnd_vector(NED_tr, [frac_TN_only, frac_NTN_only, frac_dual]);
        [~,~,~,~,~,~,Q_table_RL] = simulate_RL_baseline_offload( ...
            NED_tr, caps_tr, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
            TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
            NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
            K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
            Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
            E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
            L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
            RL_alpha, RL_gamma, eps_sched(ep), RL_num_states, RL_num_actions, ...
            Q_table_RL, true, RL_eta_loss, K_mode_loop, K_offset_dB);
    end

    for run = 1:MC_runs
        NED_max = ED_list(end);
        caps_full = mnrnd_vector(NED_max, [frac_TN_only, frac_NTN_only, frac_dual]);
        T_hat_full = max(1, round(period_mean + period_std * randn(NED_max,1)));

        for iE = 1:nE
            NED = ED_list(iE);
            caps = caps_full(1:NED);
            is_failpoint = (Nsats == 300) && (NED == 5000);
            T_hat = T_hat_full(1:NED);

            [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                tau_SINR, B_min, delta, hybrid_link_prob, Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, false, false, K_mode_loop, K_offset_dB);

            ALB_energy_runs(run,iE)=totE; ALB_delay_runs(run,iE)=avgDelay; ALB_loss_runs(run,iE)=lossProb;
            ALB_emob_runs(run,iE)=totEmob/num_epochs; ALB_off_runs(run,iE)=off_frac; ALB_lb_runs(run,iE)=lb_metric;
            ALB_delay_std_runs(run,iE)=stdDelay; ALB_delay_p95_runs(run,iE)=p95Delay; ALB_delay_p99_runs(run,iE)=p99Delay; ALB_delay_max_runs(run,iE)=maxDelay;

            [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_AHSA_epochs_offload( ...
                NED, caps, T_hat, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, capture_ratio, noise_floor, ...
                num_epochs, 0.5, 0.5, beta_adapt, r_star, p_min, p_max, Tmin_backoff, alpha_backoff, ...
                ntn_cost_alpha, ntn_prop_delay, false, false, K_mode_loop, K_offset_dB);

            AHSA_energy_runs(run,iE)=totE; AHSA_delay_runs(run,iE)=avgDelay; AHSA_loss_runs(run,iE)=lossProb;
            AHSA_emob_runs(run,iE)=totEmob/num_epochs; AHSA_off_runs(run,iE)=off_frac; AHSA_lb_runs(run,iE)=lb_metric;
            AHSA_delay_std_runs(run,iE)=stdDelay; AHSA_delay_p95_runs(run,iE)=p95Delay; AHSA_delay_p99_runs(run,iE)=p99Delay; AHSA_delay_max_runs(run,iE)=maxDelay;

            [totE, avgDelay, ~, ~, ~, ~, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                tau_SINR, B_min, delta, hybrid_link_prob, Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, true, false, K_mode_loop, K_offset_dB);

            TNonly_energy_runs(run,iE)=totE; TNonly_delay_runs(run,iE)=avgDelay; TNonly_loss_runs(run,iE)=lossProb;
            TNonly_emob_runs(run,iE)=totEmob; TNonly_off_runs(run,iE)=off_frac; TNonly_lb_runs(run,iE)=lb_metric;

            [totE, avgDelay, ~, ~, ~, ~, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                tau_SINR, B_min, delta, hybrid_link_prob, Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, false, true, K_mode_loop, K_offset_dB);

            NTNonly_energy_runs(run,iE)=totE; NTNonly_delay_runs(run,iE)=avgDelay; NTNonly_loss_runs(run,iE)=lossProb;
            NTNonly_emob_runs(run,iE)=totEmob; NTNonly_off_runs(run,iE)=off_frac; NTNonly_lb_runs(run,iE)=lb_metric;

            [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_threshold_baseline_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, TN_threshold_dB, NTN_threshold_dB, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, K_mode_loop, K_offset_dB);

            ThresholdBL_energy_runs(run,iE)=totE; ThresholdBL_delay_runs(run,iE)=avgDelay; ThresholdBL_loss_runs(run,iE)=lossProb;
            ThresholdBL_emob_runs(run,iE)=totEmob; ThresholdBL_off_runs(run,iE)=off_frac; ThresholdBL_lb_runs(run,iE)=lb_metric;

            [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric, Q_table_RL] = simulate_RL_baseline_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
                RL_alpha, RL_gamma, RL_eval_epsilon, RL_num_states, RL_num_actions, Q_table_RL, false, RL_eta_loss, K_mode_loop, K_offset_dB);

            RL_energy_runs(run,iE)=totE; RL_delay_runs(run,iE)=avgDelay; RL_loss_runs(run,iE)=lossProb;
            RL_emob_runs(run,iE)=totEmob; RL_off_runs(run,iE)=off_frac; RL_lb_runs(run,iE)=lb_metric;
        end
    end

    [ALB_energy_mean(iN,:), ALB_energy_ci(iN,:)] = mean_ci(ALB_energy_runs, z95);
    [ALB_delay_mean(iN,:),  ALB_delay_ci(iN,:)]  = mean_ci(ALB_delay_runs, z95);
    [ALB_loss_mean(iN,:),   ALB_loss_ci(iN,:)]   = mean_ci(ALB_loss_runs, z95);
    [ALB_emob_mean(iN,:),   ALB_emob_ci(iN,:)]   = mean_ci(ALB_emob_runs, z95);
    [ALB_off_mean(iN,:),    ALB_off_ci(iN,:)]    = mean_ci(ALB_off_runs, z95);
    [ALB_lb_mean(iN,:),     ALB_lb_ci(iN,:)]     = mean_ci(ALB_lb_runs, z95);

    [AHSA_energy_mean(iN,:), AHSA_energy_ci(iN,:)] = mean_ci(AHSA_energy_runs, z95);
    [AHSA_delay_mean(iN,:),  AHSA_delay_ci(iN,:)]  = mean_ci(AHSA_delay_runs, z95);
    [AHSA_loss_mean(iN,:),   AHSA_loss_ci(iN,:)]   = mean_ci(AHSA_loss_runs, z95);
    [AHSA_emob_mean(iN,:),   AHSA_emob_ci(iN,:)]   = mean_ci(AHSA_emob_runs, z95);
    [AHSA_off_mean(iN,:),    AHSA_off_ci(iN,:)]    = mean_ci(AHSA_off_runs, z95);
    [AHSA_lb_mean(iN,:),     AHSA_lb_ci(iN,:)]     = mean_ci(AHSA_lb_runs, z95);

    [TNonly_energy_mean(iN,:), TNonly_energy_ci(iN,:)] = mean_ci(TNonly_energy_runs, z95);
    [TNonly_delay_mean(iN,:),  TNonly_delay_ci(iN,:)]  = mean_ci(TNonly_delay_runs, z95);
    [TNonly_loss_mean(iN,:),   TNonly_loss_ci(iN,:)]   = mean_ci(TNonly_loss_runs, z95);

    [NTNonly_energy_mean(iN,:), NTNonly_energy_ci(iN,:)] = mean_ci(NTNonly_energy_runs, z95);
    [NTNonly_delay_mean(iN,:),  NTNonly_delay_ci(iN,:)]  = mean_ci(NTNonly_delay_runs, z95);
    [NTNonly_loss_mean(iN,:),   NTNonly_loss_ci(iN,:)]   = mean_ci(NTNonly_loss_runs, z95);
    [NTNonly_off_mean(iN,:),    NTNonly_off_ci(iN,:)]    = mean_ci(NTNonly_off_runs, z95);
    [NTNonly_lb_mean(iN,:),     NTNonly_lb_ci(iN,:)]     = mean_ci(NTNonly_lb_runs, z95);

    [ThresholdBL_energy_mean(iN,:), ThresholdBL_energy_ci(iN,:)] = mean_ci(ThresholdBL_energy_runs, z95);
    [ThresholdBL_delay_mean(iN,:),  ThresholdBL_delay_ci(iN,:)]  = mean_ci(ThresholdBL_delay_runs, z95);
    [ThresholdBL_loss_mean(iN,:),   ThresholdBL_loss_ci(iN,:)]   = mean_ci(ThresholdBL_loss_runs, z95);

    [RL_energy_mean(iN,:), RL_energy_ci(iN,:)] = mean_ci(RL_energy_runs, z95);
    [RL_delay_mean(iN,:),  RL_delay_ci(iN,:)]  = mean_ci(RL_delay_runs, z95);
    [RL_loss_mean(iN,:),   RL_loss_ci(iN,:)]   = mean_ci(RL_loss_runs, z95);
    [RL_off_mean(iN,:),    RL_off_ci(iN,:)]    = mean_ci(RL_off_runs, z95);
    [RL_lb_mean(iN,:),     RL_lb_ci(iN,:)]     = mean_ci(RL_lb_runs, z95);

    for iE = 1:nE
        [~, p_energy(iN,iE), ~, eff_energy(iN,iE), ~] = statistical_test(ALB_energy_runs(:,iE), AHSA_energy_runs(:,iE), 0.05);
        [~, p_delay(iN,iE),  ~, eff_delay(iN,iE),  ~] = statistical_test(ALB_delay_runs(:,iE),  AHSA_delay_runs(:,iE),  0.05);
        [~, p_loss(iN,iE),   ~, eff_loss(iN,iE),   ~] = statistical_test(ALB_loss_runs(:,iE),   AHSA_loss_runs(:,iE),   0.05);
    end
end

%% ==================== CLEANUP: Force regeneration of Rayleigh/Markov files ====================
if mk_enable
    % Delete stale .mat files so they are regenerated fresh
    if isfile('results_rayleigh_static_TN_NTN.mat')
        delete('results_rayleigh_static_TN_NTN.mat');
        fprintf('Deleted stale results_rayleigh_static_TN_NTN.mat\n');
    end
    if isfile('results_rayleigh_accuracy_checkpoint.mat')
        delete('results_rayleigh_accuracy_checkpoint.mat');
        fprintf('Deleted stale results_rayleigh_accuracy_checkpoint.mat\n');
    end
end

%% ==================== RAYLEIGH (K=0) MC + MARKOV ARRAYS FILL (FINAL) ====================
if mk_enable
    rng(rng_seed);

    K_factor_TN_ray  = 1;   % 0 dB
    K_factor_NTN_ray = 1;   % 0 dB
    K_mode_ray = 'scalar';
    K_offset_ray_dB = 0;

    sim_delay_TN  = zeros(nN,nE); sim_loss_TN  = zeros(nN,nE);
    sim_delay_NTN = zeros(nN,nE); sim_loss_NTN = zeros(nN,nE);

    markov_delay_TN  = nan(nN,nE); markov_loss_TN  = nan(nN,nE);
    markov_delay_NTN = nan(nN,nE); markov_loss_NTN = nan(nN,nE);

    for iN_ray = 1:nN
        Nsats_ray = Nsats_list(iN_ray);
        p_ntn_cover_ray = min(0.999999, Nsats_ray/300);
        cap_NTN_ray = sat_capacity_per_sat * Nsats_ray;

        Tslot_mean_runs = zeros(MC_runs,nE);

        ray_delay_runs_TN  = zeros(MC_runs,nE);
        ray_loss_runs_TN   = zeros(MC_runs,nE);
        ray_delay_runs_NTN = zeros(MC_runs,nE);
        ray_loss_runs_NTN  = zeros(MC_runs,nE);

        for run_ray = 1:MC_runs
            NED_max_ray = ED_list(end);
            caps_full_ray = mnrnd_vector(NED_max_ray, [frac_TN_only, frac_NTN_only, frac_dual]);
            T_hat_full_ray = max(1, round(period_mean_rayleigh + period_std_rayleigh * randn(NED_max_ray,1)));

            for iE_ray = 1:nE
                NED_ray = ED_list(iE_ray);
                caps_ray = caps_full_ray(1:NED_ray);
                T_hat_ray = T_hat_full_ray(1:NED_ray);

                % 1) TN-only static simulation
                [~, avgDelay_tn, ~, ~, ~, ~, lossProb_tn, ~, ~, ~, ~] = simulate_ALB_one_epoch_offload( ...
                    NED_ray, caps_ray, Nsats_ray, p_ntn_cover_ray, cap_NTN_ray, TN_capacity, ...
                    TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                    NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                    K_factor_TN_ray, K_factor_NTN_ray, ...
                    detection_slope, detection_threshold, ...
                    tau_SINR, B_min, delta, hybrid_link_prob, ...
                    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                    L_preambles, ntn_cost_alpha, ntn_prop_delay, true, false, K_mode_ray, K_offset_ray_dB);

                % 2) NTN-only static simulation
                fprintf('[STATIC_NTN_CALL] M=%d N=%d\n', Nsats_ray, NED_ray);
                [~, avgDelay_ntn, ~, ~, ~, ~, lossProb_ntn, ~, ~, ~, dbg_ntn_static] = simulate_ALB_one_epoch_offload( ...
                    NED_ray, caps_ray, Nsats_ray, p_ntn_cover_ray, cap_NTN_ray, TN_capacity, ...
                    TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
                    NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                    K_factor_TN_ray, K_factor_NTN_ray, ...
                    detection_slope, detection_threshold, ...
                    tau_SINR, B_min, delta, hybrid_link_prob, ...
                    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                    L_preambles, ntn_cost_alpha, ntn_prop_delay, false, true, K_mode_ray, K_offset_ray_dB);

                if Nsats_ray == 300 && (NED_ray == 500 || NED_ray == 5000)
                    fprintf('[DBG_NTN_STATIC2] NED=%d lossProb=%.4f ntn_try=%d ntn_accept=%d reject_gate=%d reject_batt=%d fallback_tn=%d buffer_entries=%d buffer_drop=%d\n', ...
                        NED_ray, lossProb_ntn, dbg_ntn_static.ntn_try, dbg_ntn_static.ntn_accept, ...
                        dbg_ntn_static.ntn_reject_gate, dbg_ntn_static.ntn_reject_batt, ...
                        dbg_ntn_static.fallback_tn, dbg_ntn_static.buffer_entries, dbg_ntn_static.buffer_drop);
                end

                ray_delay_runs_TN(run_ray,iE_ray)  = avgDelay_tn;
                ray_loss_runs_TN(run_ray,iE_ray)   = lossProb_tn;
                ray_delay_runs_NTN(run_ray,iE_ray) = avgDelay_ntn;
                ray_loss_runs_NTN(run_ray,iE_ray)  = lossProb_ntn;

                Tslot_mean_runs(run_ray,iE_ray) = mean(T_hat_ray);

            end
        end % end run_ray loop

        sim_delay_TN(iN_ray,:)  = mean(ray_delay_runs_TN,1);
        sim_loss_TN(iN_ray,:)   = mean(ray_loss_runs_TN,1);
        sim_delay_NTN(iN_ray,:) = mean(ray_delay_runs_NTN,1);
        sim_loss_NTN(iN_ray,:)  = mean(ray_loss_runs_NTN,1);

        % === Empirical NTN calibration from MC at lightest load (ED_list(1)=50) ===
        L_N50_NTN = sim_loss_NTN(iN_ray, 1);   % measured MC loss at N=50
        L_N50_NTN = max(0, min(1, L_N50_NTN)); % safety clamp
        Ps_NTN_eff = 1 - L_N50_NTN^(1/max_attempts);
        Ps_NTN_eff = max(0, min(1, Ps_NTN_eff)); % safety clamp

        for iE_ray = 1:nE
            NED_ray = ED_list(iE_ray);
            T_slots_eff = max(1, mean(Tslot_mean_runs(:,iE_ray)));

            % Build and solve share one try/catch
            try
                mk = build_markov_params_strict_v2( ...
                    NED_ray, Nsats_ray, TN_capacity, cap_NTN_ray, ...
                    L_preambles, detection_slope, detection_threshold, TN_SINR_mu, NTN_SINR_mu, ...
                    mk_lambda_mode, T_slots_eff, ...
                    mk_T_OFF_mode, mk_T_OFF_fixed_slots, p_ntn_cover_ray, ...
                    D_max, max_attempts, mk_service_mode, ...
                    mk_use_sg_pd, mk_sigma2, mk_alpha, mk_rhoPPP, mk_Omega_TN, mk_Omega_NTN, ...
                    E_tx_TN, E_tx_NTN, tau_SINR, TN_pathloss_offset_dB, TN_SINR_sigma, NTN_pathloss_offset_dB, NTN_SINR_sigma, ...
                    Ps_NTN_eff);

                fprintf('DBG M=%d N=%d: T_slots_eff=%.6f lambda=%.6f q=%.6f\n', ...
                    Nsats_ray, NED_ray, T_slots_eff, mk.lambda, mk.q);

                % TN-static Markov reference
                [Dmk_tn, Lmk_tn, pi_stat_tn, conv_tn, it_tn] = markov_model_predict_sparse_eq10_eq11_v2( ...
                    mk, 'tn_always_on', mk_pi_tol, mk_pi_maxit, mk_pi_require_convergence);

                % NTN-static Markov reference
                [Dmk_ntn, Lmk_ntn, pi_stat_ntn, conv_ntn, it_ntn] = markov_model_predict_sparse_eq10_eq11_v2( ...
                    mk, 'ntn_always_on', mk_pi_tol, mk_pi_maxit, mk_pi_require_convergence);

                fprintf('M=%d N=%d: MC_loss_TN=%.4f Markov_loss_TN=%.4f q=%.4f\n', ...
                     Nsats_ray, NED_ray, mean(ray_loss_runs_TN(:,iE_ray)), Lmk_tn, mk.q);
                
                fprintf('M=%d N=%d: MC_loss_NTN=%.4f Markov_loss_NTN=%.4f q=%.4f\n', ...
                    Nsats_ray, NED_ray, mean(ray_loss_runs_NTN(:,iE_ray)), Lmk_ntn, mk.q);

                fprintf('TN  M=%d N=%d: conv=%d it=%d sum(pi)=%.6f min(pi)=%.3e max(pi)=%.3e D=%.6f L=%.6f\n', ...
                    Nsats_ray, NED_ray, conv_tn, it_tn, sum(pi_stat_tn), min(pi_stat_tn), max(pi_stat_tn), Dmk_tn, Lmk_tn);
                fprintf('NTN M=%d N=%d: conv=%d it=%d sum(pi)=%.6f min(pi)=%.3e max(pi)=%.3e D=%.6f L=%.6f\n', ...
                    Nsats_ray, NED_ray, conv_ntn, it_ntn, sum(pi_stat_ntn), min(pi_stat_ntn), max(pi_stat_ntn), Dmk_ntn, Lmk_ntn);

            catch ME_mk
                warning('Markov build/solve failed at M=%d, N=%d: %s', Nsats_ray, NED_ray, ME_mk.message);
                Dmk_tn = NaN;  Lmk_tn = NaN;
                Dmk_ntn = NaN; Lmk_ntn = NaN;
            end
            markov_delay_TN(iN_ray,iE_ray)  = Dmk_tn;
            markov_loss_TN(iN_ray,iE_ray)   = Lmk_tn;
            markov_delay_NTN(iN_ray,iE_ray) = Dmk_ntn;
            markov_loss_NTN(iN_ray,iE_ray)  = Lmk_ntn;
        end

        % Checkpoint after each Nsats_ray so Monte Carlo results are not lost
        save('results_rayleigh_accuracy_checkpoint.mat', ...
            'sim_delay_TN','sim_loss_TN','markov_delay_TN','markov_loss_TN', ...
            'sim_delay_NTN','sim_loss_NTN','markov_delay_NTN','markov_loss_NTN', ...
            'Nsats_list','ED_list','iN_ray');
    end
end

% save canonical outputs
results_plot = build_results_snapshot( ...
    ALB_energy_mean, ALB_delay_mean, ALB_loss_mean, ALB_off_mean, ...
    AHSA_energy_mean, AHSA_delay_mean, AHSA_loss_mean, AHSA_off_mean, ...
    TNonly_energy_mean, TNonly_delay_mean, TNonly_loss_mean, ...
    NTNonly_energy_mean, NTNonly_delay_mean, NTNonly_loss_mean, ...
    ThresholdBL_energy_mean, ThresholdBL_delay_mean, ThresholdBL_loss_mean, ...
    RL_energy_mean, RL_delay_mean, RL_loss_mean, ...
    Nsats_list, ED_list, K_factor_TN_dB, K_factor_NTN_dB);


% Guard: make sure results_plot holds the canonical (K_NTN = 3 dB) arrays
if ~isfield(results_plot,'ALB_energy_mean')
    if exist('results_sensitivity','var')
        [~,idx_nom] = min(abs(K_factor_NTN_dB_sweep - K_factor_NTN_dB));
        results_plot = results_sensitivity(idx_nom);
    else
        error('results_plot is empty: canonical run did not populate outputs.');
    end
end

% Restore plotting arrays to canonical baseline (K_NTN=3 dB)
ALB_energy_mean = results_plot.ALB_energy_mean;
ALB_delay_mean  = results_plot.ALB_delay_mean;
ALB_loss_mean   = results_plot.ALB_loss_mean;
ALB_off_mean    = results_plot.ALB_off_mean;

AHSA_energy_mean = results_plot.AHSA_energy_mean;
AHSA_delay_mean  = results_plot.AHSA_delay_mean;
AHSA_loss_mean   = results_plot.AHSA_loss_mean;
AHSA_off_mean    = results_plot.AHSA_off_mean;

TNonly_energy_mean = results_plot.TNonly_energy_mean;
TNonly_delay_mean  = results_plot.TNonly_delay_mean;
TNonly_loss_mean   = results_plot.TNonly_loss_mean;

NTNonly_energy_mean = results_plot.NTNonly_energy_mean;
NTNonly_delay_mean  = results_plot.NTNonly_delay_mean;
NTNonly_loss_mean   = results_plot.NTNonly_loss_mean;

ThresholdBL_energy_mean = results_plot.ThresholdBL_energy_mean;
ThresholdBL_delay_mean  = results_plot.ThresholdBL_delay_mean;
ThresholdBL_loss_mean   = results_plot.ThresholdBL_loss_mean;

RL_energy_mean = results_plot.RL_energy_mean;
RL_delay_mean  = results_plot.RL_delay_mean;
RL_loss_mean   = results_plot.RL_loss_mean;

results_rayleigh_accuracy = results_plot;
if mk_enable
    results_rayleigh_accuracy.sim_delay_TN    = sim_delay_TN;
    results_rayleigh_accuracy.sim_loss_TN     = sim_loss_TN;
    results_rayleigh_accuracy.markov_delay_TN = markov_delay_TN;
    results_rayleigh_accuracy.markov_loss_TN  = markov_loss_TN;

    results_rayleigh_accuracy.sim_delay_NTN    = sim_delay_NTN;
    results_rayleigh_accuracy.sim_loss_NTN     = sim_loss_NTN;
    results_rayleigh_accuracy.markov_delay_NTN = markov_delay_NTN;
    results_rayleigh_accuracy.markov_loss_NTN  = markov_loss_NTN;
end

if mk_enable
    save('results_rayleigh_static_TN_NTN.mat','results_rayleigh_accuracy','Nsats_list','ED_list', ...
        'sim_delay_TN','sim_loss_TN','markov_delay_TN','markov_loss_TN', ...
        'sim_delay_NTN','sim_loss_NTN','markov_delay_NTN','markov_loss_NTN');
end

%% ==================== EXTRA MAIN-COMPARISON RUN (M=100,200) ====================
% NOTE:
% - This block is intentionally separate from the canonical [12,300] run.
% - It computes ONLY the missing main-comparison densities (M=100,200),
%   including paired stats (ATLB vs AHSA), without touching sensitivity/Rayleigh logic.

nN_extra = numel(Nsats_list_extra);

% ---------- EXTRA mean/CI containers (same metrics as the main figures) ----------
ALB_energy_mean_extra = zeros(nN_extra,nE); ALB_energy_ci_extra = zeros(nN_extra,nE);
ALB_delay_mean_extra  = zeros(nN_extra,nE); ALB_delay_ci_extra  = zeros(nN_extra,nE);
ALB_loss_mean_extra   = zeros(nN_extra,nE); ALB_loss_ci_extra   = zeros(nN_extra,nE);
ALB_emob_mean_extra   = zeros(nN_extra,nE); ALB_emob_ci_extra   = zeros(nN_extra,nE);
ALB_off_mean_extra    = zeros(nN_extra,nE); ALB_off_ci_extra    = zeros(nN_extra,nE);
ALB_lb_mean_extra     = zeros(nN_extra,nE); ALB_lb_ci_extra     = zeros(nN_extra,nE);

ALB_delay_std_mean_extra = zeros(nN_extra,nE); ALB_delay_std_ci_extra = zeros(nN_extra,nE);
ALB_delay_p95_mean_extra = zeros(nN_extra,nE); ALB_delay_p95_ci_extra = zeros(nN_extra,nE);
ALB_delay_p99_mean_extra = zeros(nN_extra,nE); ALB_delay_p99_ci_extra = zeros(nN_extra,nE);
ALB_delay_max_mean_extra = zeros(nN_extra,nE); ALB_delay_max_ci_extra = zeros(nN_extra,nE);

AHSA_energy_mean_extra = zeros(nN_extra,nE); AHSA_energy_ci_extra = zeros(nN_extra,nE);
AHSA_delay_mean_extra  = zeros(nN_extra,nE); AHSA_delay_ci_extra  = zeros(nN_extra,nE);
AHSA_loss_mean_extra   = zeros(nN_extra,nE); AHSA_loss_ci_extra   = zeros(nN_extra,nE);
AHSA_emob_mean_extra   = zeros(nN_extra,nE); AHSA_emob_ci_extra   = zeros(nN_extra,nE);
AHSA_off_mean_extra    = zeros(nN_extra,nE); AHSA_off_ci_extra    = zeros(nN_extra,nE);
AHSA_lb_mean_extra     = zeros(nN_extra,nE); AHSA_lb_ci_extra     = zeros(nN_extra,nE);

AHSA_delay_std_mean_extra = zeros(nN_extra,nE); AHSA_delay_std_ci_extra = zeros(nN_extra,nE);
AHSA_delay_p95_mean_extra = zeros(nN_extra,nE); AHSA_delay_p95_ci_extra = zeros(nN_extra,nE);
AHSA_delay_p99_mean_extra = zeros(nN_extra,nE); AHSA_delay_p99_ci_extra = zeros(nN_extra,nE);
AHSA_delay_max_mean_extra = zeros(nN_extra,nE); AHSA_delay_max_ci_extra = zeros(nN_extra,nE);


TNonly_energy_mean_extra = zeros(nN_extra,nE); TNonly_energy_ci_extra = zeros(nN_extra,nE);
TNonly_delay_mean_extra  = zeros(nN_extra,nE); TNonly_delay_ci_extra  = zeros(nN_extra,nE);
TNonly_loss_mean_extra   = zeros(nN_extra,nE); TNonly_loss_ci_extra   = zeros(nN_extra,nE);
TNonly_emob_mean_extra   = zeros(nN_extra,nE); TNonly_emob_ci_extra   = zeros(nN_extra,nE);
TNonly_off_mean_extra    = zeros(nN_extra,nE); TNonly_off_ci_extra    = zeros(nN_extra,nE);
TNonly_lb_mean_extra     = zeros(nN_extra,nE); TNonly_lb_ci_extra     = zeros(nN_extra,nE);

NTNonly_energy_mean_extra = zeros(nN_extra,nE); NTNonly_energy_ci_extra = zeros(nN_extra,nE);
NTNonly_delay_mean_extra  = zeros(nN_extra,nE); NTNonly_delay_ci_extra  = zeros(nN_extra,nE);
NTNonly_loss_mean_extra   = zeros(nN_extra,nE); NTNonly_loss_ci_extra   = zeros(nN_extra,nE);
NTNonly_emob_mean_extra   = zeros(nN_extra,nE); NTNonly_emob_ci_extra   = zeros(nN_extra,nE);
NTNonly_off_mean_extra    = zeros(nN_extra,nE); NTNonly_off_ci_extra    = zeros(nN_extra,nE);
NTNonly_lb_mean_extra     = zeros(nN_extra,nE); NTNonly_lb_ci_extra     = zeros(nN_extra,nE);

ThresholdBL_energy_mean_extra = zeros(nN_extra,nE); ThresholdBL_energy_ci_extra = zeros(nN_extra,nE);
ThresholdBL_delay_mean_extra  = zeros(nN_extra,nE); ThresholdBL_delay_ci_extra  = zeros(nN_extra,nE);
ThresholdBL_loss_mean_extra   = zeros(nN_extra,nE); ThresholdBL_loss_ci_extra   = zeros(nN_extra,nE);
ThresholdBL_emob_mean_extra   = zeros(nN_extra,nE); ThresholdBL_emob_ci_extra   = zeros(nN_extra,nE);
ThresholdBL_off_mean_extra    = zeros(nN_extra,nE); ThresholdBL_off_ci_extra    = zeros(nN_extra,nE);
ThresholdBL_lb_mean_extra     = zeros(nN_extra,nE); ThresholdBL_lb_ci_extra     = zeros(nN_extra,nE);

RL_energy_mean_extra = zeros(nN_extra,nE); RL_energy_ci_extra = zeros(nN_extra,nE);
RL_delay_mean_extra  = zeros(nN_extra,nE); RL_delay_ci_extra  = zeros(nN_extra,nE);
RL_loss_mean_extra   = zeros(nN_extra,nE); RL_loss_ci_extra   = zeros(nN_extra,nE);
RL_emob_mean_extra   = zeros(nN_extra,nE); RL_emob_ci_extra   = zeros(nN_extra,nE);
RL_off_mean_extra    = zeros(nN_extra,nE); RL_off_ci_extra    = zeros(nN_extra,nE);
RL_lb_mean_extra     = zeros(nN_extra,nE); RL_lb_ci_extra     = zeros(nN_extra,nE);

% ---------- EXTRA paired-test containers (ATLB vs AHSA) ----------
p_energy_extra = nan(nN_extra, nE);
p_delay_extra  = nan(nN_extra, nE);
p_loss_extra   = nan(nN_extra, nE);
eff_energy_extra = nan(nN_extra, nE);
eff_delay_extra  = nan(nN_extra, nE);
eff_loss_extra   = nan(nN_extra, nE);

fprintf('Running EXTRA main-comparison simulation for Nsats_list_extra = [%s] ...\n', num2str(Nsats_list_extra));

for iX = 1:nN_extra
    Nsats = Nsats_list_extra(iX);
    p_ntn_cover = min(1, Nsats/300);
    cap_NTN = sat_capacity_per_sat * Nsats;

    % ----- per-run storage (needed for mean/CI + paired statistical tests) -----
    ALB_energy_runs = zeros(MC_runs,nE); ALB_delay_runs  = zeros(MC_runs,nE); ALB_loss_runs   = zeros(MC_runs,nE);
    ALB_emob_runs   = zeros(MC_runs,nE); ALB_off_runs    = zeros(MC_runs,nE); ALB_lb_runs     = zeros(MC_runs,nE);
    ALB_delay_std_runs = zeros(MC_runs,nE); ALB_delay_p95_runs = zeros(MC_runs,nE);
    ALB_delay_p99_runs = zeros(MC_runs,nE); ALB_delay_max_runs = zeros(MC_runs,nE);

    AHSA_energy_runs = zeros(MC_runs,nE); AHSA_delay_runs  = zeros(MC_runs,nE); AHSA_loss_runs   = zeros(MC_runs,nE);
    AHSA_emob_runs   = zeros(MC_runs,nE); AHSA_off_runs    = zeros(MC_runs,nE); AHSA_lb_runs     = zeros(MC_runs,nE);
    AHSA_delay_std_runs = zeros(MC_runs,nE); AHSA_delay_p95_runs = zeros(MC_runs,nE);
    AHSA_delay_p99_runs = zeros(MC_runs,nE); AHSA_delay_max_runs = zeros(MC_runs,nE);

    TNonly_energy_runs = zeros(MC_runs,nE); TNonly_delay_runs = zeros(MC_runs,nE); TNonly_loss_runs = zeros(MC_runs,nE);
    TNonly_emob_runs   = zeros(MC_runs,nE); TNonly_off_runs   = zeros(MC_runs,nE); TNonly_lb_runs   = zeros(MC_runs,nE);

    NTNonly_energy_runs = zeros(MC_runs,nE); NTNonly_delay_runs = zeros(MC_runs,nE); NTNonly_loss_runs = zeros(MC_runs,nE);
    NTNonly_emob_runs   = zeros(MC_runs,nE); NTNonly_off_runs   = zeros(MC_runs,nE); NTNonly_lb_runs   = zeros(MC_runs,nE);

    ThresholdBL_energy_runs = zeros(MC_runs,nE); ThresholdBL_delay_runs = zeros(MC_runs,nE); ThresholdBL_loss_runs = zeros(MC_runs,nE);
    ThresholdBL_emob_runs   = zeros(MC_runs,nE); ThresholdBL_off_runs   = zeros(MC_runs,nE); ThresholdBL_lb_runs   = zeros(MC_runs,nE);

    RL_energy_runs = zeros(MC_runs,nE); RL_delay_runs = zeros(MC_runs,nE); RL_loss_runs = zeros(MC_runs,nE);
    RL_emob_runs   = zeros(MC_runs,nE); RL_off_runs   = zeros(MC_runs,nE); RL_lb_runs   = zeros(MC_runs,nE);

    % ==================== RL warm-up per density (critical) ====================
    Q_table_RL = zeros(RL_num_states, RL_num_actions);
    eps_sched = linspace(0.30, 0.02, RL_train_episodes);
    for ep = 1:RL_train_episodes
        NED_tr = ED_list(randi(nE));
        caps_tr = mnrnd_vector(NED_tr, [frac_TN_only, frac_NTN_only, frac_dual]);
        [~,~,~,~,~,~,Q_table_RL] = simulate_RL_baseline_offload( ...
            NED_tr, caps_tr, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
            TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, ...
            NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
            K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
            Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
            E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
            L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
            RL_alpha, RL_gamma, eps_sched(ep), RL_num_states, RL_num_actions, ...
            Q_table_RL, true, RL_eta_loss, K_mode_loop, K_offset_dB);
    end

    % ==================== MC evaluation ====================
    for run = 1:MC_runs
        NED_max = ED_list(end);
        caps_full = mnrnd_vector(NED_max, [frac_TN_only, frac_NTN_only, frac_dual]);
        T_hat_full = max(1, round(period_mean + period_std * randn(NED_max,1)));

        for iE = 1:nE
            NED = ED_list(iE);
            caps = caps_full(1:NED);
            T_hat = T_hat_full(1:NED);

            % ATLB
            [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                tau_SINR, B_min, delta, hybrid_link_prob, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, false, false, K_mode_loop, K_offset_dB);
            ALB_energy_runs(run,iE)=totE; ALB_delay_runs(run,iE)=avgDelay; ALB_loss_runs(run,iE)=lossProb;
            ALB_emob_runs(run,iE)=totEmob/num_epochs; ALB_off_runs(run,iE)=off_frac; ALB_lb_runs(run,iE)=lb_metric;
            ALB_delay_std_runs(run,iE)=stdDelay; ALB_delay_p95_runs(run,iE)=p95Delay;
            ALB_delay_p99_runs(run,iE)=p99Delay; ALB_delay_max_runs(run,iE)=maxDelay;

            % AHSA
            [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_AHSA_epochs_offload( ...
                NED, caps, T_hat, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, capture_ratio, noise_floor, ...
                num_epochs, 0.5, 0.5, beta_adapt, r_star, p_min, p_max, Tmin_backoff, alpha_backoff, ...
                ntn_cost_alpha, ntn_prop_delay, false, false, K_mode_loop, K_offset_dB);
            AHSA_energy_runs(run,iE)=totE; AHSA_delay_runs(run,iE)=avgDelay; AHSA_loss_runs(run,iE)=lossProb;
            AHSA_emob_runs(run,iE)=totEmob/num_epochs; AHSA_off_runs(run,iE)=off_frac; AHSA_lb_runs(run,iE)=lb_metric;
            AHSA_delay_std_runs(run,iE)=stdDelay; AHSA_delay_p95_runs(run,iE)=p95Delay;
            AHSA_delay_p99_runs(run,iE)=p99Delay; AHSA_delay_max_runs(run,iE)=maxDelay;

            % TN-only
            [totE, avgDelay, ~, ~, ~, ~, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                tau_SINR, B_min, delta, hybrid_link_prob, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, true, false, K_mode_loop, K_offset_dB);
            TNonly_energy_runs(run,iE)=totE; TNonly_delay_runs(run,iE)=avgDelay; TNonly_loss_runs(run,iE)=lossProb;
            TNonly_emob_runs(run,iE)=totEmob; TNonly_off_runs(run,iE)=off_frac; TNonly_lb_runs(run,iE)=lb_metric;

            % NTN-only
            [totE, avgDelay, ~, ~, ~, ~, lossProb, totEmob, off_frac, lb_metric, ~] = simulate_ALB_one_epoch_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                tau_SINR, B_min, delta, hybrid_link_prob, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, false, true, K_mode_loop, K_offset_dB);
            NTNonly_energy_runs(run,iE)=totE; NTNonly_delay_runs(run,iE)=avgDelay; NTNonly_loss_runs(run,iE)=lossProb;
            NTNonly_emob_runs(run,iE)=totEmob; NTNonly_off_runs(run,iE)=off_frac; NTNonly_lb_runs(run,iE)=lb_metric;

            % Threshold
            [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_threshold_baseline_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, TN_threshold_dB, NTN_threshold_dB, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, K_mode_loop, K_offset_dB);
            ThresholdBL_energy_runs(run,iE)=totE; ThresholdBL_delay_runs(run,iE)=avgDelay; ThresholdBL_loss_runs(run,iE)=lossProb;
            ThresholdBL_emob_runs(run,iE)=totEmob; ThresholdBL_off_runs(run,iE)=off_frac; ThresholdBL_lb_runs(run,iE)=lb_metric;

            % Q-learning eval (frozen warm-started Q table)
            [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric, Q_table_RL] = simulate_RL_baseline_offload( ...
                NED, caps, Nsats, p_ntn_cover, cap_NTN, TN_capacity, ...
                TN_SINR_mu, TN_SINR_sigma, TN_pathloss_offset_dB, NTN_pathloss_offset_dB, NTN_SINR_mu, NTN_SINR_sigma, ...
                K_factor_TN, K_factor_NTN, detection_slope, detection_threshold, ...
                Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
                E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
                L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
                RL_alpha, RL_gamma, RL_eval_epsilon, RL_num_states, RL_num_actions, Q_table_RL, false, RL_eta_loss, K_mode_loop, K_offset_dB);
            RL_energy_runs(run,iE)=totE; RL_delay_runs(run,iE)=avgDelay; RL_loss_runs(run,iE)=lossProb;
            RL_emob_runs(run,iE)=totEmob; RL_off_runs(run,iE)=off_frac; RL_lb_runs(run,iE)=lb_metric;
        end
    end

    % ----- mean/CI for plotting -----
    [ALB_energy_mean_extra(iX,:), ALB_energy_ci_extra(iX,:)] = mean_ci(ALB_energy_runs, z95);
    [ALB_delay_mean_extra(iX,:),  ALB_delay_ci_extra(iX,:)]  = mean_ci(ALB_delay_runs, z95);
    [ALB_loss_mean_extra(iX,:),   ALB_loss_ci_extra(iX,:)]   = mean_ci(ALB_loss_runs, z95);
    [ALB_emob_mean_extra(iX,:),   ALB_emob_ci_extra(iX,:)]   = mean_ci(ALB_emob_runs, z95);
    [ALB_off_mean_extra(iX,:),    ALB_off_ci_extra(iX,:)]    = mean_ci(ALB_off_runs, z95);
    [ALB_lb_mean_extra(iX,:),     ALB_lb_ci_extra(iX,:)]     = mean_ci(ALB_lb_runs, z95);

    % ALB delay distribution stats
    [ALB_delay_std_mean_extra(iX,:), ALB_delay_std_ci_extra(iX,:)] = mean_ci(ALB_delay_std_runs, z95);
    [ALB_delay_p95_mean_extra(iX,:), ALB_delay_p95_ci_extra(iX,:)] = mean_ci(ALB_delay_p95_runs, z95);
    [ALB_delay_p99_mean_extra(iX,:), ALB_delay_p99_ci_extra(iX,:)] = mean_ci(ALB_delay_p99_runs, z95);
    [ALB_delay_max_mean_extra(iX,:), ALB_delay_max_ci_extra(iX,:)] = mean_ci(ALB_delay_max_runs, z95);

    [AHSA_energy_mean_extra(iX,:), AHSA_energy_ci_extra(iX,:)] = mean_ci(AHSA_energy_runs, z95);
    [AHSA_delay_mean_extra(iX,:),  AHSA_delay_ci_extra(iX,:)]  = mean_ci(AHSA_delay_runs, z95);
    [AHSA_loss_mean_extra(iX,:),   AHSA_loss_ci_extra(iX,:)]   = mean_ci(AHSA_loss_runs, z95);
    [AHSA_emob_mean_extra(iX,:),   AHSA_emob_ci_extra(iX,:)]   = mean_ci(AHSA_emob_runs, z95);
    [AHSA_off_mean_extra(iX,:),    AHSA_off_ci_extra(iX,:)]    = mean_ci(AHSA_off_runs, z95);
    [AHSA_lb_mean_extra(iX,:),     AHSA_lb_ci_extra(iX,:)]     = mean_ci(AHSA_lb_runs, z95);

    % AHSA delay distribution stats
    [AHSA_delay_std_mean_extra(iX,:), AHSA_delay_std_ci_extra(iX,:)] = mean_ci(AHSA_delay_std_runs, z95);
    [AHSA_delay_p95_mean_extra(iX,:), AHSA_delay_p95_ci_extra(iX,:)] = mean_ci(AHSA_delay_p95_runs, z95);
    [AHSA_delay_p99_mean_extra(iX,:), AHSA_delay_p99_ci_extra(iX,:)] = mean_ci(AHSA_delay_p99_runs, z95);
    [AHSA_delay_max_mean_extra(iX,:), AHSA_delay_max_ci_extra(iX,:)] = mean_ci(AHSA_delay_max_runs, z95);

    [TNonly_energy_mean_extra(iX,:), TNonly_energy_ci_extra(iX,:)] = mean_ci(TNonly_energy_runs, z95);
    [TNonly_delay_mean_extra(iX,:),  TNonly_delay_ci_extra(iX,:)]  = mean_ci(TNonly_delay_runs, z95);
    [TNonly_loss_mean_extra(iX,:),   TNonly_loss_ci_extra(iX,:)]   = mean_ci(TNonly_loss_runs, z95);
    [TNonly_emob_mean_extra(iX,:),   TNonly_emob_ci_extra(iX,:)]   = mean_ci(TNonly_emob_runs, z95);
    [TNonly_off_mean_extra(iX,:),    TNonly_off_ci_extra(iX,:)]    = mean_ci(TNonly_off_runs, z95);
    [TNonly_lb_mean_extra(iX,:),     TNonly_lb_ci_extra(iX,:)]     = mean_ci(TNonly_lb_runs, z95);

    [NTNonly_energy_mean_extra(iX,:), NTNonly_energy_ci_extra(iX,:)] = mean_ci(NTNonly_energy_runs, z95);
    [NTNonly_delay_mean_extra(iX,:),  NTNonly_delay_ci_extra(iX,:)]  = mean_ci(NTNonly_delay_runs, z95);
    [NTNonly_loss_mean_extra(iX,:),   NTNonly_loss_ci_extra(iX,:)]   = mean_ci(NTNonly_loss_runs, z95);
    [NTNonly_emob_mean_extra(iX,:),   NTNonly_emob_ci_extra(iX,:)]   = mean_ci(NTNonly_emob_runs, z95);
    [NTNonly_off_mean_extra(iX,:),    NTNonly_off_ci_extra(iX,:)]    = mean_ci(NTNonly_off_runs, z95);
    [NTNonly_lb_mean_extra(iX,:),     NTNonly_lb_ci_extra(iX,:)]     = mean_ci(NTNonly_lb_runs, z95);

    [ThresholdBL_energy_mean_extra(iX,:), ThresholdBL_energy_ci_extra(iX,:)] = mean_ci(ThresholdBL_energy_runs, z95);
    [ThresholdBL_delay_mean_extra(iX,:),  ThresholdBL_delay_ci_extra(iX,:)]  = mean_ci(ThresholdBL_delay_runs, z95);
    [ThresholdBL_loss_mean_extra(iX,:),   ThresholdBL_loss_ci_extra(iX,:)]   = mean_ci(ThresholdBL_loss_runs, z95);
    [ThresholdBL_emob_mean_extra(iX,:),   ThresholdBL_emob_ci_extra(iX,:)]   = mean_ci(ThresholdBL_emob_runs, z95);
    [ThresholdBL_off_mean_extra(iX,:),    ThresholdBL_off_ci_extra(iX,:)]    = mean_ci(ThresholdBL_off_runs, z95);
    [ThresholdBL_lb_mean_extra(iX,:),     ThresholdBL_lb_ci_extra(iX,:)]     = mean_ci(ThresholdBL_lb_runs, z95);

    [RL_energy_mean_extra(iX,:), RL_energy_ci_extra(iX,:)] = mean_ci(RL_energy_runs, z95);
    [RL_delay_mean_extra(iX,:),  RL_delay_ci_extra(iX,:)]  = mean_ci(RL_delay_runs, z95);
    [RL_loss_mean_extra(iX,:),   RL_loss_ci_extra(iX,:)]   = mean_ci(RL_loss_runs, z95);
    [RL_emob_mean_extra(iX,:),   RL_emob_ci_extra(iX,:)]   = mean_ci(RL_emob_runs, z95);
    [RL_off_mean_extra(iX,:),    RL_off_ci_extra(iX,:)]    = mean_ci(RL_off_runs, z95);
    [RL_lb_mean_extra(iX,:),     RL_lb_ci_extra(iX,:)]     = mean_ci(RL_lb_runs, z95);

    % ----- paired tests for Table 5 (ATLB vs AHSA) -----
    for iE = 1:nE
        [~, p_energy_extra(iX,iE), ~, eff_energy_extra(iX,iE), ~] = statistical_test(ALB_energy_runs(:,iE), AHSA_energy_runs(:,iE), 0.05);
        [~, p_delay_extra(iX,iE),  ~, eff_delay_extra(iX,iE),  ~] = statistical_test(ALB_delay_runs(:,iE),  AHSA_delay_runs(:,iE),  0.05);
        [~, p_loss_extra(iX,iE),   ~, eff_loss_extra(iX,iE),   ~] = statistical_test(ALB_loss_runs(:,iE),   AHSA_loss_runs(:,iE),   0.05);
    end

    fprintf('  EXTRA completed M=%d\n', Nsats);
end

%% ==================== MERGE CANONICAL (12,300) + EXTRA (100,200) INTO MAIN-4 ARRAYS ====================

% Row indices in main list
[~,i12]  = ismember(12,  Nsats_list_main);
[~,i100] = ismember(100, Nsats_list_main);
[~,i200] = ismember(200, Nsats_list_main);
[~,i300] = ismember(300, Nsats_list_main);

assert(all([i12 i100 i200 i300] > 0), 'Nsats_list_main must contain [12 100 200 300].');

nN_main = numel(Nsats_list_main);

% -------- allocate merged arrays --------
ALB_energy_mean_main4 = nan(nN_main,nE); ALB_energy_ci_main4 = nan(nN_main,nE);
ALB_delay_mean_main4  = nan(nN_main,nE); ALB_delay_ci_main4  = nan(nN_main,nE);
ALB_loss_mean_main4   = nan(nN_main,nE); ALB_loss_ci_main4   = nan(nN_main,nE);
ALB_emob_mean_main4   = nan(nN_main,nE); ALB_emob_ci_main4   = nan(nN_main,nE);
ALB_off_mean_main4    = nan(nN_main,nE); ALB_off_ci_main4    = nan(nN_main,nE);
ALB_lb_mean_main4     = nan(nN_main,nE); ALB_lb_ci_main4     = nan(nN_main,nE);

AHSA_energy_mean_main4 = nan(nN_main,nE); AHSA_energy_ci_main4 = nan(nN_main,nE);
AHSA_delay_mean_main4  = nan(nN_main,nE); AHSA_delay_ci_main4  = nan(nN_main,nE);
AHSA_loss_mean_main4   = nan(nN_main,nE); AHSA_loss_ci_main4   = nan(nN_main,nE);
AHSA_emob_mean_main4   = nan(nN_main,nE); AHSA_emob_ci_main4   = nan(nN_main,nE);
AHSA_off_mean_main4    = nan(nN_main,nE); AHSA_off_ci_main4    = nan(nN_main,nE);
AHSA_lb_mean_main4     = nan(nN_main,nE); AHSA_lb_ci_main4     = nan(nN_main,nE);

TNonly_energy_mean_main4 = nan(nN_main,nE); TNonly_energy_ci_main4 = nan(nN_main,nE);
TNonly_delay_mean_main4  = nan(nN_main,nE); TNonly_delay_ci_main4  = nan(nN_main,nE);
TNonly_loss_mean_main4   = nan(nN_main,nE); TNonly_loss_ci_main4   = nan(nN_main,nE);
TNonly_emob_mean_main4   = nan(nN_main,nE); TNonly_emob_ci_main4   = nan(nN_main,nE);
TNonly_off_mean_main4    = nan(nN_main,nE); TNonly_off_ci_main4    = nan(nN_main,nE);
TNonly_lb_mean_main4     = nan(nN_main,nE); TNonly_lb_ci_main4     = nan(nN_main,nE);

NTNonly_energy_mean_main4 = nan(nN_main,nE); NTNonly_energy_ci_main4 = nan(nN_main,nE);
NTNonly_delay_mean_main4  = nan(nN_main,nE); NTNonly_delay_ci_main4  = nan(nN_main,nE);
NTNonly_loss_mean_main4   = nan(nN_main,nE); NTNonly_loss_ci_main4   = nan(nN_main,nE);
NTNonly_emob_mean_main4   = nan(nN_main,nE); NTNonly_emob_ci_main4   = nan(nN_main,nE);
NTNonly_off_mean_main4    = nan(nN_main,nE); NTNonly_off_ci_main4    = nan(nN_main,nE);
NTNonly_lb_mean_main4     = nan(nN_main,nE); NTNonly_lb_ci_main4     = nan(nN_main,nE);

ThresholdBL_energy_mean_main4 = nan(nN_main,nE); ThresholdBL_energy_ci_main4 = nan(nN_main,nE);
ThresholdBL_delay_mean_main4  = nan(nN_main,nE); ThresholdBL_delay_ci_main4  = nan(nN_main,nE);
ThresholdBL_loss_mean_main4   = nan(nN_main,nE); ThresholdBL_loss_ci_main4   = nan(nN_main,nE);
ThresholdBL_emob_mean_main4   = nan(nN_main,nE); ThresholdBL_emob_ci_main4   = nan(nN_main,nE);
ThresholdBL_off_mean_main4    = nan(nN_main,nE); ThresholdBL_off_ci_main4    = nan(nN_main,nE);
ThresholdBL_lb_mean_main4     = nan(nN_main,nE); ThresholdBL_lb_ci_main4     = nan(nN_main,nE);

RL_energy_mean_main4 = nan(nN_main,nE); RL_energy_ci_main4 = nan(nN_main,nE);
RL_delay_mean_main4  = nan(nN_main,nE); RL_delay_ci_main4  = nan(nN_main,nE);
RL_loss_mean_main4   = nan(nN_main,nE); RL_loss_ci_main4   = nan(nN_main,nE);
RL_emob_mean_main4   = nan(nN_main,nE); RL_emob_ci_main4   = nan(nN_main,nE);
RL_off_mean_main4    = nan(nN_main,nE); RL_off_ci_main4    = nan(nN_main,nE);
RL_lb_mean_main4     = nan(nN_main,nE); RL_lb_ci_main4     = nan(nN_main,nE);

% ALB delay distribution (main4)
ALB_delay_std_mean_main4 = nan(nN_main,nE); ALB_delay_std_ci_main4 = nan(nN_main,nE);
ALB_delay_p95_mean_main4 = nan(nN_main,nE); ALB_delay_p95_ci_main4 = nan(nN_main,nE);
ALB_delay_p99_mean_main4 = nan(nN_main,nE); ALB_delay_p99_ci_main4 = nan(nN_main,nE);
ALB_delay_max_mean_main4 = nan(nN_main,nE); ALB_delay_max_ci_main4 = nan(nN_main,nE);

% AHSA delay distribution (main4)
AHSA_delay_std_mean_main4 = nan(nN_main,nE); AHSA_delay_std_ci_main4 = nan(nN_main,nE);
AHSA_delay_p95_mean_main4 = nan(nN_main,nE); AHSA_delay_p95_ci_main4 = nan(nN_main,nE);
AHSA_delay_p99_mean_main4 = nan(nN_main,nE); AHSA_delay_p99_ci_main4 = nan(nN_main,nE);
AHSA_delay_max_mean_main4 = nan(nN_main,nE); AHSA_delay_max_ci_main4 = nan(nN_main,nE);

% stats main4
p_energy_main4   = nan(nN_main,nE);
p_delay_main4    = nan(nN_main,nE);
p_loss_main4     = nan(nN_main,nE);
eff_energy_main4 = nan(nN_main,nE);
eff_delay_main4  = nan(nN_main,nE);
eff_loss_main4   = nan(nN_main,nE);

% -------- canonical rows mapping (Nsats_list = [12,300]) --------
ALB_energy_mean_main4(i12,:) = ALB_energy_mean(1,:);   ALB_energy_ci_main4(i12,:) = ALB_energy_ci(1,:);
ALB_energy_mean_main4(i300,:) = ALB_energy_mean(2,:);  ALB_energy_ci_main4(i300,:) = ALB_energy_ci(2,:);
ALB_delay_mean_main4(i12,:) = ALB_delay_mean(1,:);     ALB_delay_ci_main4(i12,:) = ALB_delay_ci(1,:);
ALB_delay_mean_main4(i300,:) = ALB_delay_mean(2,:);    ALB_delay_ci_main4(i300,:) = ALB_delay_ci(2,:);
ALB_loss_mean_main4(i12,:) = ALB_loss_mean(1,:);       ALB_loss_ci_main4(i12,:) = ALB_loss_ci(1,:);
ALB_loss_mean_main4(i300,:) = ALB_loss_mean(2,:);      ALB_loss_ci_main4(i300,:) = ALB_loss_ci(2,:);
ALB_emob_mean_main4(i12,:) = ALB_emob_mean(1,:);       ALB_emob_ci_main4(i12,:) = ALB_emob_ci(1,:);
ALB_emob_mean_main4(i300,:) = ALB_emob_mean(2,:);      ALB_emob_ci_main4(i300,:) = ALB_emob_ci(2,:);
ALB_off_mean_main4(i12,:) = ALB_off_mean(1,:);         ALB_off_ci_main4(i12,:) = ALB_off_ci(1,:);
ALB_off_mean_main4(i300,:) = ALB_off_mean(2,:);        ALB_off_ci_main4(i300,:) = ALB_off_ci(2,:);
ALB_lb_mean_main4(i12,:) = ALB_lb_mean(1,:);           ALB_lb_ci_main4(i12,:) = ALB_lb_ci(1,:);
ALB_lb_mean_main4(i300,:) = ALB_lb_mean(2,:);          ALB_lb_ci_main4(i300,:) = ALB_lb_ci(2,:);

AHSA_energy_mean_main4(i12,:) = AHSA_energy_mean(1,:); AHSA_energy_ci_main4(i12,:) = AHSA_energy_ci(1,:);
AHSA_energy_mean_main4(i300,:) = AHSA_energy_mean(2,:);AHSA_energy_ci_main4(i300,:) = AHSA_energy_ci(2,:);
AHSA_delay_mean_main4(i12,:) = AHSA_delay_mean(1,:);   AHSA_delay_ci_main4(i12,:) = AHSA_delay_ci(1,:);
AHSA_delay_mean_main4(i300,:) = AHSA_delay_mean(2,:);  AHSA_delay_ci_main4(i300,:) = AHSA_delay_ci(2,:);
AHSA_loss_mean_main4(i12,:) = AHSA_loss_mean(1,:);     AHSA_loss_ci_main4(i12,:) = AHSA_loss_ci(1,:);
AHSA_loss_mean_main4(i300,:) = AHSA_loss_mean(2,:);    AHSA_loss_ci_main4(i300,:) = AHSA_loss_ci(2,:);
AHSA_emob_mean_main4(i12,:) = AHSA_emob_mean(1,:);     AHSA_emob_ci_main4(i12,:) = AHSA_emob_ci(1,:);
AHSA_emob_mean_main4(i300,:) = AHSA_emob_mean(2,:);    AHSA_emob_ci_main4(i300,:) = AHSA_emob_ci(2,:);
AHSA_off_mean_main4(i12,:) = AHSA_off_mean(1,:);       AHSA_off_ci_main4(i12,:) = AHSA_off_ci(1,:);
AHSA_off_mean_main4(i300,:) = AHSA_off_mean(2,:);      AHSA_off_ci_main4(i300,:) = AHSA_off_ci(2,:);
AHSA_lb_mean_main4(i12,:) = AHSA_lb_mean(1,:);         AHSA_lb_ci_main4(i12,:) = AHSA_lb_ci(1,:);
AHSA_lb_mean_main4(i300,:) = AHSA_lb_mean(2,:);        AHSA_lb_ci_main4(i300,:) = AHSA_lb_ci(2,:);

% ALB canonical delay distribution rows
ALB_delay_std_mean_main4(i12,:) = ALB_delay_std_mean(1,:); ALB_delay_std_ci_main4(i12,:) = ALB_delay_std_ci(1,:);
ALB_delay_std_mean_main4(i300,:)= ALB_delay_std_mean(2,:); ALB_delay_std_ci_main4(i300,:)= ALB_delay_std_ci(2,:);

ALB_delay_p95_mean_main4(i12,:) = ALB_delay_p95_mean(1,:); ALB_delay_p95_ci_main4(i12,:) = ALB_delay_p95_ci(1,:);
ALB_delay_p95_mean_main4(i300,:)= ALB_delay_p95_mean(2,:); ALB_delay_p95_ci_main4(i300,:)= ALB_delay_p95_ci(2,:);

ALB_delay_p99_mean_main4(i12,:) = ALB_delay_p99_mean(1,:); ALB_delay_p99_ci_main4(i12,:) = ALB_delay_p99_ci(1,:);
ALB_delay_p99_mean_main4(i300,:)= ALB_delay_p99_mean(2,:); ALB_delay_p99_ci_main4(i300,:)= ALB_delay_p99_ci(2,:);

ALB_delay_max_mean_main4(i12,:) = ALB_delay_max_mean(1,:); ALB_delay_max_ci_main4(i12,:) = ALB_delay_max_ci(1,:);
ALB_delay_max_mean_main4(i300,:)= ALB_delay_max_mean(2,:); ALB_delay_max_ci_main4(i300,:)= ALB_delay_max_ci(2,:);

% AHSA canonical delay distribution rows
AHSA_delay_std_mean_main4(i12,:) = AHSA_delay_std_mean(1,:); AHSA_delay_std_ci_main4(i12,:) = AHSA_delay_std_ci(1,:);
AHSA_delay_std_mean_main4(i300,:)= AHSA_delay_std_mean(2,:); AHSA_delay_std_ci_main4(i300,:)= AHSA_delay_std_ci(2,:);

AHSA_delay_p95_mean_main4(i12,:) = AHSA_delay_p95_mean(1,:); AHSA_delay_p95_ci_main4(i12,:) = AHSA_delay_p95_ci(1,:);
AHSA_delay_p95_mean_main4(i300,:)= AHSA_delay_p95_mean(2,:); AHSA_delay_p95_ci_main4(i300,:)= AHSA_delay_p95_ci(2,:);

AHSA_delay_p99_mean_main4(i12,:) = AHSA_delay_p99_mean(1,:); AHSA_delay_p99_ci_main4(i12,:) = AHSA_delay_p99_ci(1,:);
AHSA_delay_p99_mean_main4(i300,:)= AHSA_delay_p99_mean(2,:); AHSA_delay_p99_ci_main4(i300,:)= AHSA_delay_p99_ci(2,:);

AHSA_delay_max_mean_main4(i12,:) = AHSA_delay_max_mean(1,:); AHSA_delay_max_ci_main4(i12,:) = AHSA_delay_max_ci(1,:);
AHSA_delay_max_mean_main4(i300,:)= AHSA_delay_max_mean(2,:); AHSA_delay_max_ci_main4(i300,:)= AHSA_delay_max_ci(2,:);


TNonly_energy_mean_main4(i12,:) = TNonly_energy_mean(1,:); TNonly_energy_ci_main4(i12,:) = TNonly_energy_ci(1,:);
TNonly_energy_mean_main4(i300,:) = TNonly_energy_mean(2,:);TNonly_energy_ci_main4(i300,:) = TNonly_energy_ci(2,:);
TNonly_delay_mean_main4(i12,:) = TNonly_delay_mean(1,:);   TNonly_delay_ci_main4(i12,:) = TNonly_delay_ci(1,:);
TNonly_delay_mean_main4(i300,:) = TNonly_delay_mean(2,:);  TNonly_delay_ci_main4(i300,:) = TNonly_delay_ci(2,:);
TNonly_loss_mean_main4(i12,:) = TNonly_loss_mean(1,:);     TNonly_loss_ci_main4(i12,:) = TNonly_loss_ci(1,:);
TNonly_loss_mean_main4(i300,:) = TNonly_loss_mean(2,:);    TNonly_loss_ci_main4(i300,:) = TNonly_loss_ci(2,:);
TNonly_emob_mean_main4(i12,:) = TNonly_emob_mean(1,:);     TNonly_emob_ci_main4(i12,:) = TNonly_emob_ci(1,:);
TNonly_emob_mean_main4(i300,:) = TNonly_emob_mean(2,:);    TNonly_emob_ci_main4(i300,:) = TNonly_emob_ci(2,:);
TNonly_off_mean_main4(i12,:) = TNonly_off_mean(1,:);       TNonly_off_ci_main4(i12,:) = TNonly_off_ci(1,:);
TNonly_off_mean_main4(i300,:) = TNonly_off_mean(2,:);      TNonly_off_ci_main4(i300,:) = TNonly_off_ci(2,:);
TNonly_lb_mean_main4(i12,:) = TNonly_lb_mean(1,:);         TNonly_lb_ci_main4(i12,:) = TNonly_lb_ci(1,:);
TNonly_lb_mean_main4(i300,:) = TNonly_lb_mean(2,:);        TNonly_lb_ci_main4(i300,:) = TNonly_lb_ci(2,:);

NTNonly_energy_mean_main4(i12,:) = NTNonly_energy_mean(1,:); NTNonly_energy_ci_main4(i12,:) = NTNonly_energy_ci(1,:);
NTNonly_energy_mean_main4(i300,:) = NTNonly_energy_mean(2,:);NTNonly_energy_ci_main4(i300,:) = NTNonly_energy_ci(2,:);
NTNonly_delay_mean_main4(i12,:) = NTNonly_delay_mean(1,:);   NTNonly_delay_ci_main4(i12,:) = NTNonly_delay_ci(1,:);
NTNonly_delay_mean_main4(i300,:) = NTNonly_delay_mean(2,:);  NTNonly_delay_ci_main4(i300,:) = NTNonly_delay_ci(2,:);
NTNonly_loss_mean_main4(i12,:) = NTNonly_loss_mean(1,:);     NTNonly_loss_ci_main4(i12,:) = NTNonly_loss_ci(1,:);
NTNonly_loss_mean_main4(i300,:) = NTNonly_loss_mean(2,:);    NTNonly_loss_ci_main4(i300,:) = NTNonly_loss_ci(2,:);
NTNonly_emob_mean_main4(i12,:) = NTNonly_emob_mean(1,:);     NTNonly_emob_ci_main4(i12,:) = NTNonly_emob_ci(1,:);
NTNonly_emob_mean_main4(i300,:) = NTNonly_emob_mean(2,:);    NTNonly_emob_ci_main4(i300,:) = NTNonly_emob_ci(2,:);
NTNonly_off_mean_main4(i12,:) = NTNonly_off_mean(1,:);       NTNonly_off_ci_main4(i12,:) = NTNonly_off_ci(1,:);
NTNonly_off_mean_main4(i300,:) = NTNonly_off_mean(2,:);      NTNonly_off_ci_main4(i300,:) = NTNonly_off_ci(2,:);
NTNonly_lb_mean_main4(i12,:) = NTNonly_lb_mean(1,:);         NTNonly_lb_ci_main4(i12,:) = NTNonly_lb_ci(1,:);
NTNonly_lb_mean_main4(i300,:) = NTNonly_lb_mean(2,:);        NTNonly_lb_ci_main4(i300,:) = NTNonly_lb_ci(2,:);

ThresholdBL_energy_mean_main4(i12,:) = ThresholdBL_energy_mean(1,:); ThresholdBL_energy_ci_main4(i12,:) = ThresholdBL_energy_ci(1,:);
ThresholdBL_energy_mean_main4(i300,:) = ThresholdBL_energy_mean(2,:);ThresholdBL_energy_ci_main4(i300,:) = ThresholdBL_energy_ci(2,:);
ThresholdBL_delay_mean_main4(i12,:) = ThresholdBL_delay_mean(1,:);   ThresholdBL_delay_ci_main4(i12,:) = ThresholdBL_delay_ci(1,:);
ThresholdBL_delay_mean_main4(i300,:) = ThresholdBL_delay_mean(2,:);  ThresholdBL_delay_ci_main4(i300,:) = ThresholdBL_delay_ci(2,:);
ThresholdBL_loss_mean_main4(i12,:) = ThresholdBL_loss_mean(1,:);     ThresholdBL_loss_ci_main4(i12,:) = ThresholdBL_loss_ci(1,:);
ThresholdBL_loss_mean_main4(i300,:) = ThresholdBL_loss_mean(2,:);    ThresholdBL_loss_ci_main4(i300,:) = ThresholdBL_loss_ci(2,:);
ThresholdBL_emob_mean_main4(i12,:) = ThresholdBL_emob_mean(1,:);     ThresholdBL_emob_ci_main4(i12,:) = ThresholdBL_emob_ci(1,:);
ThresholdBL_emob_mean_main4(i300,:) = ThresholdBL_emob_mean(2,:);    ThresholdBL_emob_ci_main4(i300,:) = ThresholdBL_emob_ci(2,:);
ThresholdBL_off_mean_main4(i12,:) = ThresholdBL_off_mean(1,:);       ThresholdBL_off_ci_main4(i12,:) = ThresholdBL_off_ci(1,:);
ThresholdBL_off_mean_main4(i300,:) = ThresholdBL_off_mean(2,:);      ThresholdBL_off_ci_main4(i300,:) = ThresholdBL_off_ci(2,:);
ThresholdBL_lb_mean_main4(i12,:) = ThresholdBL_lb_mean(1,:);         ThresholdBL_lb_ci_main4(i12,:) = ThresholdBL_lb_ci(1,:);
ThresholdBL_lb_mean_main4(i300,:) = ThresholdBL_lb_mean(2,:);        ThresholdBL_lb_ci_main4(i300,:) = ThresholdBL_lb_ci(2,:);

RL_energy_mean_main4(i12,:) = RL_energy_mean(1,:); RL_energy_ci_main4(i12,:) = RL_energy_ci(1,:);
RL_energy_mean_main4(i300,:) = RL_energy_mean(2,:);RL_energy_ci_main4(i300,:) = RL_energy_ci(2,:);
RL_delay_mean_main4(i12,:) = RL_delay_mean(1,:);   RL_delay_ci_main4(i12,:) = RL_delay_ci(1,:);
RL_delay_mean_main4(i300,:) = RL_delay_mean(2,:);  RL_delay_ci_main4(i300,:) = RL_delay_ci(2,:);
RL_loss_mean_main4(i12,:) = RL_loss_mean(1,:);     RL_loss_ci_main4(i12,:) = RL_loss_ci(1,:);
RL_loss_mean_main4(i300,:) = RL_loss_mean(2,:);    RL_loss_ci_main4(i300,:) = RL_loss_ci(2,:);
RL_emob_mean_main4(i12,:) = RL_emob_mean(1,:);     RL_emob_ci_main4(i12,:) = RL_emob_ci(1,:);
RL_emob_mean_main4(i300,:) = RL_emob_mean(2,:);    RL_emob_ci_main4(i300,:) = RL_emob_ci(2,:);
RL_off_mean_main4(i12,:) = RL_off_mean(1,:);       RL_off_ci_main4(i12,:) = RL_off_ci(1,:);
RL_off_mean_main4(i300,:) = RL_off_mean(2,:);      RL_off_ci_main4(i300,:) = RL_off_ci(2,:);
RL_lb_mean_main4(i12,:) = RL_lb_mean(1,:);         RL_lb_ci_main4(i12,:) = RL_lb_ci(1,:);
RL_lb_mean_main4(i300,:) = RL_lb_mean(2,:);        RL_lb_ci_main4(i300,:) = RL_lb_ci(2,:);

% canonical stats (rows 1->M12, 2->M300 in existing arrays)
p_energy_main4(i12,:)   = p_energy(1,:);   p_energy_main4(i300,:)   = p_energy(2,:);
p_delay_main4(i12,:)    = p_delay(1,:);    p_delay_main4(i300,:)    = p_delay(2,:);
p_loss_main4(i12,:)     = p_loss(1,:);     p_loss_main4(i300,:)     = p_loss(2,:);
eff_energy_main4(i12,:) = eff_energy(1,:); eff_energy_main4(i300,:) = eff_energy(2,:);
eff_delay_main4(i12,:)  = eff_delay(1,:);  eff_delay_main4(i300,:)  = eff_delay(2,:);
eff_loss_main4(i12,:)   = eff_loss(1,:);   eff_loss_main4(i300,:)   = eff_loss(2,:);

% -------- extra rows mapping (Nsats_list_extra = [100,200], iX=1,2) --------
ALB_energy_mean_main4(i100,:) = ALB_energy_mean_extra(1,:); ALB_energy_ci_main4(i100,:) = ALB_energy_ci_extra(1,:);
ALB_energy_mean_main4(i200,:) = ALB_energy_mean_extra(2,:); ALB_energy_ci_main4(i200,:) = ALB_energy_ci_extra(2,:);
ALB_delay_mean_main4(i100,:)  = ALB_delay_mean_extra(1,:);  ALB_delay_ci_main4(i100,:)  = ALB_delay_ci_extra(1,:);
ALB_delay_mean_main4(i200,:)  = ALB_delay_mean_extra(2,:);  ALB_delay_ci_main4(i200,:)  = ALB_delay_ci_extra(2,:);
ALB_loss_mean_main4(i100,:)   = ALB_loss_mean_extra(1,:);   ALB_loss_ci_main4(i100,:)   = ALB_loss_ci_extra(1,:);
ALB_loss_mean_main4(i200,:)   = ALB_loss_mean_extra(2,:);   ALB_loss_ci_main4(i200,:)   = ALB_loss_ci_extra(2,:);
ALB_emob_mean_main4(i100,:)   = ALB_emob_mean_extra(1,:);   ALB_emob_ci_main4(i100,:)   = ALB_emob_ci_extra(1,:);
ALB_emob_mean_main4(i200,:)   = ALB_emob_mean_extra(2,:);   ALB_emob_ci_main4(i200,:)   = ALB_emob_ci_extra(2,:);
ALB_off_mean_main4(i100,:)    = ALB_off_mean_extra(1,:);    ALB_off_ci_main4(i100,:)    = ALB_off_ci_extra(1,:);
ALB_off_mean_main4(i200,:)    = ALB_off_mean_extra(2,:);    ALB_off_ci_main4(i200,:)    = ALB_off_ci_extra(2,:);
ALB_lb_mean_main4(i100,:)     = ALB_lb_mean_extra(1,:);     ALB_lb_ci_main4(i100,:)     = ALB_lb_ci_extra(1,:);
ALB_lb_mean_main4(i200,:)     = ALB_lb_mean_extra(2,:);     ALB_lb_ci_main4(i200,:)     = ALB_lb_ci_extra(2,:);

AHSA_energy_mean_main4(i100,:) = AHSA_energy_mean_extra(1,:); AHSA_energy_ci_main4(i100,:) = AHSA_energy_ci_extra(1,:);
AHSA_energy_mean_main4(i200,:) = AHSA_energy_mean_extra(2,:); AHSA_energy_ci_main4(i200,:) = AHSA_energy_ci_extra(2,:);
AHSA_delay_mean_main4(i100,:)  = AHSA_delay_mean_extra(1,:);  AHSA_delay_ci_main4(i100,:)  = AHSA_delay_ci_extra(1,:);
AHSA_delay_mean_main4(i200,:)  = AHSA_delay_mean_extra(2,:);  AHSA_delay_ci_main4(i200,:)  = AHSA_delay_ci_extra(2,:);
AHSA_loss_mean_main4(i100,:)   = AHSA_loss_mean_extra(1,:);   AHSA_loss_ci_main4(i100,:)   = AHSA_loss_ci_extra(1,:);
AHSA_loss_mean_main4(i200,:)   = AHSA_loss_mean_extra(2,:);   AHSA_loss_ci_main4(i200,:)   = AHSA_loss_ci_extra(2,:);
AHSA_emob_mean_main4(i100,:)   = AHSA_emob_mean_extra(1,:);   AHSA_emob_ci_main4(i100,:)   = AHSA_emob_ci_extra(1,:);
AHSA_emob_mean_main4(i200,:)   = AHSA_emob_mean_extra(2,:);   AHSA_emob_ci_main4(i200,:)   = AHSA_emob_ci_extra(2,:);
AHSA_off_mean_main4(i100,:)    = AHSA_off_mean_extra(1,:);    AHSA_off_ci_main4(i100,:)    = AHSA_off_ci_extra(1,:);
AHSA_off_mean_main4(i200,:)    = AHSA_off_mean_extra(2,:);    AHSA_off_ci_main4(i200,:)    = AHSA_off_ci_extra(2,:);
AHSA_lb_mean_main4(i100,:)     = AHSA_lb_mean_extra(1,:);     AHSA_lb_ci_main4(i100,:)     = AHSA_lb_ci_extra(1,:);
AHSA_lb_mean_main4(i200,:)     = AHSA_lb_mean_extra(2,:);     AHSA_lb_ci_main4(i200,:)     = AHSA_lb_ci_extra(2,:);

% ALB extra delay distribution rows
ALB_delay_std_mean_main4(i100,:) = ALB_delay_std_mean_extra(1,:); ALB_delay_std_ci_main4(i100,:) = ALB_delay_std_ci_extra(1,:);
ALB_delay_std_mean_main4(i200,:) = ALB_delay_std_mean_extra(2,:); ALB_delay_std_ci_main4(i200,:) = ALB_delay_std_ci_extra(2,:);

ALB_delay_p95_mean_main4(i100,:) = ALB_delay_p95_mean_extra(1,:); ALB_delay_p95_ci_main4(i100,:) = ALB_delay_p95_ci_extra(1,:);
ALB_delay_p95_mean_main4(i200,:) = ALB_delay_p95_mean_extra(2,:); ALB_delay_p95_ci_main4(i200,:) = ALB_delay_p95_ci_extra(2,:);

ALB_delay_p99_mean_main4(i100,:) = ALB_delay_p99_mean_extra(1,:); ALB_delay_p99_ci_main4(i100,:) = ALB_delay_p99_ci_extra(1,:);
ALB_delay_p99_mean_main4(i200,:) = ALB_delay_p99_mean_extra(2,:); ALB_delay_p99_ci_main4(i200,:) = ALB_delay_p99_ci_extra(2,:);

ALB_delay_max_mean_main4(i100,:) = ALB_delay_max_mean_extra(1,:); ALB_delay_max_ci_main4(i100,:) = ALB_delay_max_ci_extra(1,:);
ALB_delay_max_mean_main4(i200,:) = ALB_delay_max_mean_extra(2,:); ALB_delay_max_ci_main4(i200,:) = ALB_delay_max_ci_extra(2,:);

% AHSA extra delay distribution rows
AHSA_delay_std_mean_main4(i100,:) = AHSA_delay_std_mean_extra(1,:); AHSA_delay_std_ci_main4(i100,:) = AHSA_delay_std_ci_extra(1,:);
AHSA_delay_std_mean_main4(i200,:) = AHSA_delay_std_mean_extra(2,:); AHSA_delay_std_ci_main4(i200,:) = AHSA_delay_std_ci_extra(2,:);

AHSA_delay_p95_mean_main4(i100,:) = AHSA_delay_p95_mean_extra(1,:); AHSA_delay_p95_ci_main4(i100,:) = AHSA_delay_p95_ci_extra(1,:);
AHSA_delay_p95_mean_main4(i200,:) = AHSA_delay_p95_mean_extra(2,:); AHSA_delay_p95_ci_main4(i200,:) = AHSA_delay_p95_ci_extra(2,:);

AHSA_delay_p99_mean_main4(i100,:) = AHSA_delay_p99_mean_extra(1,:); AHSA_delay_p99_ci_main4(i100,:) = AHSA_delay_p99_ci_extra(1,:);
AHSA_delay_p99_mean_main4(i200,:) = AHSA_delay_p99_mean_extra(2,:); AHSA_delay_p99_ci_main4(i200,:) = AHSA_delay_p99_ci_extra(2,:);

AHSA_delay_max_mean_main4(i100,:) = AHSA_delay_max_mean_extra(1,:); AHSA_delay_max_ci_main4(i100,:) = AHSA_delay_max_ci_extra(1,:);
AHSA_delay_max_mean_main4(i200,:) = AHSA_delay_max_mean_extra(2,:); AHSA_delay_max_ci_main4(i200,:) = AHSA_delay_max_ci_extra(2,:);


TNonly_energy_mean_main4(i100,:) = TNonly_energy_mean_extra(1,:); TNonly_energy_ci_main4(i100,:) = TNonly_energy_ci_extra(1,:);
TNonly_energy_mean_main4(i200,:) = TNonly_energy_mean_extra(2,:); TNonly_energy_ci_main4(i200,:) = TNonly_energy_ci_extra(2,:);
TNonly_delay_mean_main4(i100,:)  = TNonly_delay_mean_extra(1,:);  TNonly_delay_ci_main4(i100,:)  = TNonly_delay_ci_extra(1,:);
TNonly_delay_mean_main4(i200,:)  = TNonly_delay_mean_extra(2,:);  TNonly_delay_ci_main4(i200,:)  = TNonly_delay_ci_extra(2,:);
TNonly_loss_mean_main4(i100,:)   = TNonly_loss_mean_extra(1,:);   TNonly_loss_ci_main4(i100,:)   = TNonly_loss_ci_extra(1,:);
TNonly_loss_mean_main4(i200,:)   = TNonly_loss_mean_extra(2,:);   TNonly_loss_ci_main4(i200,:)   = TNonly_loss_ci_extra(2,:);
TNonly_emob_mean_main4(i100,:)   = TNonly_emob_mean_extra(1,:);   TNonly_emob_ci_main4(i100,:)   = TNonly_emob_ci_extra(1,:);
TNonly_emob_mean_main4(i200,:)   = TNonly_emob_mean_extra(2,:);   TNonly_emob_ci_main4(i200,:)   = TNonly_emob_ci_extra(2,:);
TNonly_off_mean_main4(i100,:)    = TNonly_off_mean_extra(1,:);    TNonly_off_ci_main4(i100,:)    = TNonly_off_ci_extra(1,:);
TNonly_off_mean_main4(i200,:)    = TNonly_off_mean_extra(2,:);    TNonly_off_ci_main4(i200,:)    = TNonly_off_ci_extra(2,:);
TNonly_lb_mean_main4(i100,:)     = TNonly_lb_mean_extra(1,:);     TNonly_lb_ci_main4(i100,:)     = TNonly_lb_ci_extra(1,:);
TNonly_lb_mean_main4(i200,:)     = TNonly_lb_mean_extra(2,:);     TNonly_lb_ci_main4(i200,:)     = TNonly_lb_ci_extra(2,:);

NTNonly_energy_mean_main4(i100,:) = NTNonly_energy_mean_extra(1,:); NTNonly_energy_ci_main4(i100,:) = NTNonly_energy_ci_extra(1,:);
NTNonly_energy_mean_main4(i200,:) = NTNonly_energy_mean_extra(2,:); NTNonly_energy_ci_main4(i200,:) = NTNonly_energy_ci_extra(2,:);
NTNonly_delay_mean_main4(i100,:)  = NTNonly_delay_mean_extra(1,:);  NTNonly_delay_ci_main4(i100,:)  = NTNonly_delay_ci_extra(1,:);
NTNonly_delay_mean_main4(i200,:)  = NTNonly_delay_mean_extra(2,:);  NTNonly_delay_ci_main4(i200,:)  = NTNonly_delay_ci_extra(2,:);
NTNonly_loss_mean_main4(i100,:)   = NTNonly_loss_mean_extra(1,:);   NTNonly_loss_ci_main4(i100,:)   = NTNonly_loss_ci_extra(1,:);
NTNonly_loss_mean_main4(i200,:)   = NTNonly_loss_mean_extra(2,:);   NTNonly_loss_ci_main4(i200,:)   = NTNonly_loss_ci_extra(2,:);
NTNonly_emob_mean_main4(i100,:)   = NTNonly_emob_mean_extra(1,:);   NTNonly_emob_ci_main4(i100,:)   = NTNonly_emob_ci_extra(1,:);
NTNonly_emob_mean_main4(i200,:)   = NTNonly_emob_mean_extra(2,:);   NTNonly_emob_ci_main4(i200,:)   = NTNonly_emob_ci_extra(2,:);
NTNonly_off_mean_main4(i100,:)    = NTNonly_off_mean_extra(1,:);    NTNonly_off_ci_main4(i100,:)    = NTNonly_off_ci_extra(1,:);
NTNonly_off_mean_main4(i200,:)    = NTNonly_off_mean_extra(2,:);    NTNonly_off_ci_main4(i200,:)    = NTNonly_off_ci_extra(2,:);
NTNonly_lb_mean_main4(i100,:)     = NTNonly_lb_mean_extra(1,:);     NTNonly_lb_ci_main4(i100,:)     = NTNonly_lb_ci_extra(1,:);
NTNonly_lb_mean_main4(i200,:)     = NTNonly_lb_mean_extra(2,:);     NTNonly_lb_ci_main4(i200,:)     = NTNonly_lb_ci_extra(2,:);

ThresholdBL_energy_mean_main4(i100,:) = ThresholdBL_energy_mean_extra(1,:); ThresholdBL_energy_ci_main4(i100,:) = ThresholdBL_energy_ci_extra(1,:);
ThresholdBL_energy_mean_main4(i200,:) = ThresholdBL_energy_mean_extra(2,:); ThresholdBL_energy_ci_main4(i200,:) = ThresholdBL_energy_ci_extra(2,:);
ThresholdBL_delay_mean_main4(i100,:)  = ThresholdBL_delay_mean_extra(1,:);  ThresholdBL_delay_ci_main4(i100,:)  = ThresholdBL_delay_ci_extra(1,:);
ThresholdBL_delay_mean_main4(i200,:)  = ThresholdBL_delay_mean_extra(2,:);  ThresholdBL_delay_ci_main4(i200,:)  = ThresholdBL_delay_ci_extra(2,:);
ThresholdBL_loss_mean_main4(i100,:)   = ThresholdBL_loss_mean_extra(1,:);   ThresholdBL_loss_ci_main4(i100,:)   = ThresholdBL_loss_ci_extra(1,:);
ThresholdBL_loss_mean_main4(i200,:)   = ThresholdBL_loss_mean_extra(2,:);   ThresholdBL_loss_ci_main4(i200,:)   = ThresholdBL_loss_ci_extra(2,:);
ThresholdBL_emob_mean_main4(i100,:)   = ThresholdBL_emob_mean_extra(1,:);   ThresholdBL_emob_ci_main4(i100,:)   = ThresholdBL_emob_ci_extra(1,:);
ThresholdBL_emob_mean_main4(i200,:)   = ThresholdBL_emob_mean_extra(2,:);   ThresholdBL_emob_ci_main4(i200,:)   = ThresholdBL_emob_ci_extra(2,:);
ThresholdBL_off_mean_main4(i100,:)    = ThresholdBL_off_mean_extra(1,:);    ThresholdBL_off_ci_main4(i100,:)    = ThresholdBL_off_ci_extra(1,:);
ThresholdBL_off_mean_main4(i200,:)    = ThresholdBL_off_mean_extra(2,:);    ThresholdBL_off_ci_main4(i200,:)    = ThresholdBL_off_ci_extra(2,:);
ThresholdBL_lb_mean_main4(i100,:)     = ThresholdBL_lb_mean_extra(1,:);     ThresholdBL_lb_ci_main4(i100,:)     = ThresholdBL_lb_ci_extra(1,:);
ThresholdBL_lb_mean_main4(i200,:)     = ThresholdBL_lb_mean_extra(2,:);     ThresholdBL_lb_ci_main4(i200,:)     = ThresholdBL_lb_ci_extra(2,:);

RL_energy_mean_main4(i100,:) = RL_energy_mean_extra(1,:); RL_energy_ci_main4(i100,:) = RL_energy_ci_extra(1,:);
RL_energy_mean_main4(i200,:) = RL_energy_mean_extra(2,:); RL_energy_ci_main4(i200,:) = RL_energy_ci_extra(2,:);
RL_delay_mean_main4(i100,:)  = RL_delay_mean_extra(1,:);  RL_delay_ci_main4(i100,:)  = RL_delay_ci_extra(1,:);
RL_delay_mean_main4(i200,:)  = RL_delay_mean_extra(2,:);  RL_delay_ci_main4(i200,:)  = RL_delay_ci_extra(2,:);
RL_loss_mean_main4(i100,:)   = RL_loss_mean_extra(1,:);   RL_loss_ci_main4(i100,:)   = RL_loss_ci_extra(1,:);
RL_loss_mean_main4(i200,:)   = RL_loss_mean_extra(2,:);   RL_loss_ci_main4(i200,:)   = RL_loss_ci_extra(2,:);
RL_emob_mean_main4(i100,:)   = RL_emob_mean_extra(1,:);   RL_emob_ci_main4(i100,:)   = RL_emob_ci_extra(1,:);
RL_emob_mean_main4(i200,:)   = RL_emob_mean_extra(2,:);   RL_emob_ci_main4(i200,:)   = RL_emob_ci_extra(2,:);
RL_off_mean_main4(i100,:)    = RL_off_mean_extra(1,:);    RL_off_ci_main4(i100,:)    = RL_off_ci_extra(1,:);
RL_off_mean_main4(i200,:)    = RL_off_mean_extra(2,:);    RL_off_ci_main4(i200,:)    = RL_off_ci_extra(2,:);
RL_lb_mean_main4(i100,:)     = RL_lb_mean_extra(1,:);     RL_lb_ci_main4(i100,:)     = RL_lb_ci_extra(1,:);
RL_lb_mean_main4(i200,:)     = RL_lb_mean_extra(2,:);     RL_lb_ci_main4(i200,:)     = RL_lb_ci_extra(2,:);

% extra stats
p_energy_main4(i100,:)   = p_energy_extra(1,:);   p_energy_main4(i200,:)   = p_energy_extra(2,:);
p_delay_main4(i100,:)    = p_delay_extra(1,:);    p_delay_main4(i200,:)    = p_delay_extra(2,:);
p_loss_main4(i100,:)     = p_loss_extra(1,:);     p_loss_main4(i200,:)     = p_loss_extra(2,:);
eff_energy_main4(i100,:) = eff_energy_extra(1,:); eff_energy_main4(i200,:) = eff_energy_extra(2,:);
eff_delay_main4(i100,:)  = eff_delay_extra(1,:);  eff_delay_main4(i200,:)  = eff_delay_extra(2,:);
eff_loss_main4(i100,:)   = eff_loss_extra(1,:);   eff_loss_main4(i200,:)   = eff_loss_extra(2,:);

fprintf('Merged main comparison arrays prepared for M = [%s]\n', num2str(Nsats_list_main));

save('results_main4_full.mat');

save('results.mat');

%% ==================== SENSITIVITY + RAYLEIGH-MATCHED CONSISTENCY FIGURES ====================
if export_sensitivity_figs
    % --- Sensitivity figures (need the sensitivity results) ---
    if exist('results_sensitivity_KNTN.mat','file')==2
        S = load('results_sensitivity_KNTN.mat','results_sensitivity','Nsats_list','ED_list');

        Kvals = arrayfun(@(x) x.K_factor_NTN_dB, S.results_sensitivity);
        [Kvals, ordK] = sort(Kvals);
        Ns_list = S.Nsats_list;
        EDs = S.ED_list;

        make_K_sensitivity_figure(S.results_sensitivity, ordK, Kvals, Ns_list, EDs, 'energy');
        make_K_sensitivity_figure(S.results_sensitivity, ordK, Kvals, Ns_list, EDs, 'delay');
        make_K_sensitivity_figure(S.results_sensitivity, ordK, Kvals, Ns_list, EDs, 'loss');
    else
        warning('results_sensitivity_KNTN.mat not found. Sensitivity figures skipped.');
        Ns_list = Nsats_list;   % fallback for rayleigh plots below
        EDs = ED_list;
    end

    % --- Rayleigh matched figures ---
    if exist('results_rayleigh_static_TN_NTN.mat','file')==2
        R = load('results_rayleigh_static_TN_NTN.mat','results_rayleigh_accuracy','Nsats_list','ED_list');
        RR = R.results_rayleigh_accuracy;
        req = {'sim_delay_TN','sim_loss_TN','markov_delay_TN','markov_loss_TN', ...
            'sim_delay_NTN','sim_loss_NTN','markov_delay_NTN','markov_loss_NTN'};
        for k = 1:numel(req)
            if ~isfield(RR, req{k}), error('Missing field: %s', req{k}); end
        end
    else
        RR = results_plot;
    end

    haveMarkovTN  = isfield(RR,'markov_delay_TN') && isfield(RR,'markov_loss_TN') && ...
        ~isempty(RR.markov_delay_TN) && ~isempty(RR.markov_loss_TN) && ...
        any(isfinite(RR.markov_delay_TN(:))) && any(isfinite(RR.markov_loss_TN(:)));

    haveMarkovNTN = isfield(RR,'markov_delay_NTN') && isfield(RR,'markov_loss_NTN') && ...
        ~isempty(RR.markov_delay_NTN) && ~isempty(RR.markov_loss_NTN) && ...
        any(isfinite(RR.markov_delay_NTN(:))) && any(isfinite(RR.markov_loss_NTN(:)));

    fprintf('[Debug] haveMarkovTN=%d, haveMarkovNTN=%d\n', haveMarkovTN, haveMarkovNTN);

if isfield(RR,'markov_delay_TN')
    fprintf('[Debug] TN finite delay=%d, finite loss=%d\n', ...
        nnz(isfinite(RR.markov_delay_TN(:))), nnz(isfinite(RR.markov_loss_TN(:))));
end
if isfield(RR,'markov_delay_NTN')
    fprintf('[Debug] NTN finite delay=%d, finite loss=%d\n', ...
        nnz(isfinite(RR.markov_delay_NTN(:))), nnz(isfinite(RR.markov_loss_NTN(:))));
end

% Warn instead of silently skipping the plots
if ~haveMarkovTN && ~haveMarkovNTN
    warning('Both TN and NTN Markov arrays are invalid (all-NaN or empty). Rayleigh plots 13-16 cannot be generated.');
end

    if haveMarkovTN && haveMarkovNTN
        make_rayleigh_match_figure_combined(EDs, Ns_list, ...
            RR.sim_delay_TN, RR.markov_delay_TN, RR.sim_delay_NTN, RR.markov_delay_NTN, ...
            RR.sim_loss_TN,  RR.markov_loss_TN,  RR.sim_loss_NTN,  RR.markov_loss_NTN);
    else
        if ~haveMarkovTN
            warning('TN-only Markov arrays empty/all-NaN. Combined Rayleigh figure skipped.');
        end
        if ~haveMarkovNTN
            warning('NTN-only Markov arrays empty/all-NaN. Combined Rayleigh figure skipped.');
        end
    end
    % --- ATLB parameter sensitivity figures ---
    make_param_sensitivity_tripanel(results_delta, results_Bmin, results_tauSINR, ...
        delta_sweep, Bmin_sweep, tauSINR_sweep, EDs, Nsats_list, 1, 'delay');
    make_param_sensitivity_tripanel(results_delta, results_Bmin, results_tauSINR, ...
        delta_sweep, Bmin_sweep, tauSINR_sweep, EDs, Nsats_list, 2, 'delay');

    make_param_sensitivity_tripanel(results_delta, results_Bmin, results_tauSINR, ...
        delta_sweep, Bmin_sweep, tauSINR_sweep, EDs, Nsats_list, 1, 'offload');
    make_param_sensitivity_tripanel(results_delta, results_Bmin, results_tauSINR, ...
        delta_sweep, Bmin_sweep, tauSINR_sweep, EDs, Nsats_list, 2, 'offload');
end


%% ==================== PLOTTING ====================

assert(exist('ALB_energy_mean_main4','var')==1, 'Missing merged main4 arrays');
assert(size(ALB_energy_mean_main4,1)==nN_main, 'main4 row count mismatch');
assert(all(ismember([12 100 200 300], Nsats_list_main)), 'Nsats_list_main must be [12 100 200 300]');

set(groot,'defaultAxesFontSize', 21);
set(groot,'defaultTextFontSize', 21);
set(groot,'defaultLegendFontSize', 21);



mk_ATLB = 'o';
mk_AHSA = 's';
mk_TN   = '^';
mk_NTN  = 'd';
mk_TH   = 'v';
mk_QL   = 'p';

%%======================================================================
%% PLOT 1: Energy vs #EDs
figure('Name','Energy vs Number of EDs','Color','w','Position',[100 100 1400 1000]);
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

labs_no_M = {'ATLB','AHSA','TN-only','NTN-only','Threshold','Q-Learning'};
hLegend = gobjects(1,6);

for iN = 1:nN_main
    nexttile; hold on; grid on; box on;
    x = ED_list;
    ls='-'; mfc='auto'; lwA=2.2; lwB=1.8; ms1=7; ms2=8;

    yA = ALB_energy_mean_main4(iN,:); ciA = ALB_energy_ci_main4(iN,:);
    h1 = plot(x,yA,[ls mk_ATLB],'Color',clr_ATLB,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc);
    errorbar(x,yA,ciA,'LineStyle','none','Color',clr_ATLB);

    yH = AHSA_energy_mean_main4(iN,:); ciH = AHSA_energy_ci_main4(iN,:);
    h2 = plot(x,yH,[ls mk_AHSA],'Color',clr_AHSA,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc);
    errorbar(x,yH,ciH,'LineStyle','none','Color',clr_AHSA);

    yT = TNonly_energy_mean_main4(iN,:); ciT = TNonly_energy_ci_main4(iN,:);
    h3 = plot(x,yT,[ls mk_TN],'Color',clr_TN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc);
    errorbar(x,yT,ciT,'LineStyle','none','Color',clr_TN);

    yN = NTNonly_energy_mean_main4(iN,:); ciN = NTNonly_energy_ci_main4(iN,:);
    h4 = plot(x,yN,[ls mk_NTN],'Color',clr_NTN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc);
    errorbar(x,yN,ciN,'LineStyle','none','Color',clr_NTN);

    yTh = ThresholdBL_energy_mean_main4(iN,:); ciTh = ThresholdBL_energy_ci_main4(iN,:);
    h5 = plot(x,yTh,[ls mk_TH],'Color',clr_TH,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc);
    errorbar(x,yTh,ciTh,'LineStyle','none','Color',clr_TH);

    yRL = RL_energy_mean_main4(iN,:); ciRL = RL_energy_ci_main4(iN,:);
    h6 = plot(x,yRL,[ls mk_QL],'Color',clr_QL,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc);
    errorbar(x,yRL,ciRL,'LineStyle','none','Color',clr_QL);

    title(sprintf('M = %d', Nsats_list_main(iN)));
    set(gca,'FontSize',21,'LineWidth',1.5)

    if iN==1, hLegend=[h1 h2 h3 h4 h5 h6]; end
end
lg = legend(hLegend,labs_no_M,'Orientation','horizontal'); lg.Layout.Tile='south';
xlabel(tl,'Number of EDs','FontSize',21); ylabel(tl,'Energy [mJ]','FontSize',21);

%% PLOT 2: Delay vs #EDs
figure('Name','Delay vs Number of EDs','Color','w','Position',[100 100 1400 1000]);
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

hLegend = gobjects(1,6);
for iN = 1:nN_main
    nexttile; hold on; grid on; box on;
    x = ED_list;
    ls='-'; mfc='auto'; lwA=2.2; lwB=1.8; ms1=7; ms2=8;

    yA = ALB_delay_mean_main4(iN,:); ciA = ALB_delay_ci_main4(iN,:);
    h1 = plot(x,yA,[ls mk_ATLB],'Color',clr_ATLB,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yA,ciA,'LineStyle','none','Color',clr_ATLB);

    yH = AHSA_delay_mean_main4(iN,:); ciH = AHSA_delay_ci_main4(iN,:);
    h2 = plot(x,yH,[ls mk_AHSA],'Color',clr_AHSA,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yH,ciH,'LineStyle','none','Color',clr_AHSA);

    yT = TNonly_delay_mean_main4(iN,:); ciT = TNonly_delay_ci_main4(iN,:);
    h3 = plot(x,yT,[ls mk_TN],'Color',clr_TN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yT,ciT,'LineStyle','none','Color',clr_TN);

    yN = NTNonly_delay_mean_main4(iN,:); ciN = NTNonly_delay_ci_main4(iN,:);
    h4 = plot(x,yN,[ls mk_NTN],'Color',clr_NTN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yN,ciN,'LineStyle','none','Color',clr_NTN);

    yTh = ThresholdBL_delay_mean_main4(iN,:); ciTh = ThresholdBL_delay_ci_main4(iN,:);
    h5 = plot(x,yTh,[ls mk_TH],'Color',clr_TH,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yTh,ciTh,'LineStyle','none','Color',clr_TH);

    yRL = RL_delay_mean_main4(iN,:); ciRL = RL_delay_ci_main4(iN,:);
    h6 = plot(x,yRL,[ls mk_QL],'Color',clr_QL,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yRL,ciRL,'LineStyle','none','Color',clr_QL);

    title(sprintf('M = %d', Nsats_list_main(iN)));
    set(gca,'FontSize',21,'LineWidth',1.5)

    if iN==1, hLegend=[h1 h2 h3 h4 h5 h6]; end
end
lg = legend(hLegend,labs_no_M,'Orientation','horizontal'); lg.Layout.Tile='south';
xlabel(tl,'Number of EDs','FontSize',21); ylabel(tl,'Avg delay [slots]','FontSize',21);

%% PLOT 3: Loss vs #EDs
figure('Name','Loss vs Number of EDs','Color','w','Position',[100 100 1400 1000]);
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

hLegend = gobjects(1,6);
for iN = 1:nN_main
    nexttile; hold on; grid on; box on;
    x = ED_list;
    ls='-'; mfc='auto'; lwA=2.2; lwB=1.8; ms1=7; ms2=8;

    yA = ALB_loss_mean_main4(iN,:); ciA = ALB_loss_ci_main4(iN,:);
    h1 = plot(x,yA,[ls mk_ATLB],'Color',clr_ATLB,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yA,ciA,'LineStyle','none','Color',clr_ATLB);

    yH = AHSA_loss_mean_main4(iN,:); ciH = AHSA_loss_ci_main4(iN,:);
    h2 = plot(x,yH,[ls mk_AHSA],'Color',clr_AHSA,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yH,ciH,'LineStyle','none','Color',clr_AHSA);

    yT = TNonly_loss_mean_main4(iN,:); ciT = TNonly_loss_ci_main4(iN,:);
    h3 = plot(x,yT,[ls mk_TN],'Color',clr_TN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yT,ciT,'LineStyle','none','Color',clr_TN);

    yN = NTNonly_loss_mean_main4(iN,:); ciN = NTNonly_loss_ci_main4(iN,:);
    h4 = plot(x,yN,[ls mk_NTN],'Color',clr_NTN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yN,ciN,'LineStyle','none','Color',clr_NTN);

    yTh = ThresholdBL_loss_mean_main4(iN,:); ciTh = ThresholdBL_loss_ci_main4(iN,:);
    h5 = plot(x,yTh,[ls mk_TH],'Color',clr_TH,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yTh,ciTh,'LineStyle','none','Color',clr_TH);

    yRL = RL_loss_mean_main4(iN,:); ciRL = RL_loss_ci_main4(iN,:);
    h6 = plot(x,yRL,[ls mk_QL],'Color',clr_QL,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yRL,ciRL,'LineStyle','none','Color',clr_QL);

    title(sprintf('M = %d', Nsats_list_main(iN))); ylim([0 1]);
    set(gca,'FontSize',21,'LineWidth',1.5)

    if iN==1, hLegend=[h1 h2 h3 h4 h5 h6]; end
end
lg = legend(hLegend,labs_no_M,'Orientation','horizontal'); lg.Layout.Tile='south';
xlabel(tl,'Number of EDs','FontSize',21); ylabel(tl,'Loss probability','FontSize',21);

%% PLOT 4: Energy + Mobility vs #EDs
figure('Name','Energy + Mobility vs Number of EDs','Color','w','Position',[100 100 1400 1000]);
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

hLegend = gobjects(1,6);
for iN = 1:nN_main
    nexttile; hold on; grid on; box on;
    x = ED_list;
    ls='-'; mfc='auto'; lwA=2.2; lwB=1.8; ms1=7; ms2=8;

    yA = ALB_emob_mean_main4(iN,:); ciA = ALB_emob_ci_main4(iN,:);
    h1 = plot(x,yA,[ls mk_ATLB],'Color',clr_ATLB,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yA,ciA,'LineStyle','none','Color',clr_ATLB);

    yH = AHSA_emob_mean_main4(iN,:); ciH = AHSA_emob_ci_main4(iN,:);
    h2 = plot(x,yH,[ls mk_AHSA],'Color',clr_AHSA,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yH,ciH,'LineStyle','none','Color',clr_AHSA);

    yT = TNonly_emob_mean_main4(iN,:); ciT = TNonly_emob_ci_main4(iN,:);
    h3 = plot(x,yT,[ls mk_TN],'Color',clr_TN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yT,ciT,'LineStyle','none','Color',clr_TN);

    yN = NTNonly_emob_mean_main4(iN,:); ciN = NTNonly_emob_ci_main4(iN,:);
    h4 = plot(x,yN,[ls mk_NTN],'Color',clr_NTN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yN,ciN,'LineStyle','none','Color',clr_NTN);

    yTh = ThresholdBL_emob_mean_main4(iN,:); ciTh = ThresholdBL_emob_ci_main4(iN,:);
    h5 = plot(x,yTh,[ls mk_TH],'Color',clr_TH,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yTh,ciTh,'LineStyle','none','Color',clr_TH);

    yRL = RL_emob_mean_main4(iN,:); ciRL = RL_emob_ci_main4(iN,:);
    h6 = plot(x,yRL,[ls mk_QL],'Color',clr_QL,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yRL,ciRL,'LineStyle','none','Color',clr_QL);

    title(sprintf('M = %d', Nsats_list_main(iN)));
    set(gca,'FontSize',21,'LineWidth',1.5)

    if iN==1, hLegend=[h1 h2 h3 h4 h5 h6]; end
end
lg = legend(hLegend,labs_no_M,'Orientation','horizontal'); lg.Layout.Tile='south';
xlabel(tl,'Number of EDs','FontSize',21); ylabel(tl,'Energy + mobility [mJ]','FontSize',21);

%% PLOT 5: Offload fraction vs #EDs
figure('Name','NTN Offload Fraction vs Number of EDs','Color','w','Position',[100 100 1400 1000]);
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

hLegend = gobjects(1,6);
for iN = 1:nN_main
    nexttile; hold on; grid on; box on;
    x = ED_list;
    ls='-'; mfc='auto'; lwA=2.2; lwB=1.8; ms1=7; ms2=8;

    yA = ALB_off_mean_main4(iN,:); ciA = ALB_off_ci_main4(iN,:);
    h1 = plot(x,yA,[ls mk_ATLB],'Color',clr_ATLB,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yA,ciA,'LineStyle','none','Color',clr_ATLB);

    yH = AHSA_off_mean_main4(iN,:); ciH = AHSA_off_ci_main4(iN,:);
    h2 = plot(x,yH,[ls mk_AHSA],'Color',clr_AHSA,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yH,ciH,'LineStyle','none','Color',clr_AHSA);

    yT = TNonly_off_mean_main4(iN,:); ciT = TNonly_off_ci_main4(iN,:);
    h3 = plot(x,yT,[ls mk_TN],'Color',clr_TN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yT,ciT,'LineStyle','none','Color',clr_TN);

    yN = NTNonly_off_mean_main4(iN,:); ciN = NTNonly_off_ci_main4(iN,:);
    h4 = plot(x,yN,[ls mk_NTN],'Color',clr_NTN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yN,ciN,'LineStyle','none','Color',clr_NTN);

    yTh = ThresholdBL_off_mean_main4(iN,:); ciTh = ThresholdBL_off_ci_main4(iN,:);
    h5 = plot(x,yTh,[ls mk_TH],'Color',clr_TH,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yTh,ciTh,'LineStyle','none','Color',clr_TH);

    yRL = RL_off_mean_main4(iN,:); ciRL = RL_off_ci_main4(iN,:);
    h6 = plot(x,yRL,[ls mk_QL],'Color',clr_QL,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yRL,ciRL,'LineStyle','none','Color',clr_QL);

    title(sprintf('M = %d', Nsats_list_main(iN))); ylim([0 1]);
    set(gca,'FontSize',21,'LineWidth',1.5)

    if iN==1, hLegend=[h1 h2 h3 h4 h5 h6]; end
end
lg = legend(hLegend,labs_no_M,'Orientation','horizontal'); lg.Layout.Tile='south';
xlabel(tl,'Number of EDs','FontSize',21); ylabel(tl,'NTN usage fraction','FontSize',21);

%% PLOT 6: Load-Balancing Metric vs #EDs
figure('Name','Load-Balancing Metric vs Number of EDs','Color','w','Position',[100 100 1400 1000]);
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

hLegend = gobjects(1,6);
for iN = 1:nN_main
    nexttile; hold on; grid on; box on;
    x = ED_list;
    ls='-'; mfc='auto'; lwA=2.2; lwB=1.8; ms1=7; ms2=8;

    yA = ALB_lb_mean_main4(iN,:); ciA = ALB_lb_ci_main4(iN,:);
    h1 = plot(x,yA,[ls mk_ATLB],'Color',clr_ATLB,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yA,ciA,'LineStyle','none','Color',clr_ATLB);

    yH = AHSA_lb_mean_main4(iN,:); ciH = AHSA_lb_ci_main4(iN,:);
    h2 = plot(x,yH,[ls mk_AHSA],'Color',clr_AHSA,'LineWidth',lwA,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yH,ciH,'LineStyle','none','Color',clr_AHSA);

    yT = TNonly_lb_mean_main4(iN,:); ciT = TNonly_lb_ci_main4(iN,:);
    h3 = plot(x,yT,[ls mk_TN],'Color',clr_TN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yT,ciT,'LineStyle','none','Color',clr_TN);

    yN = NTNonly_lb_mean_main4(iN,:); ciN = NTNonly_lb_ci_main4(iN,:);
    h4 = plot(x,yN,[ls mk_NTN],'Color',clr_NTN,'LineWidth',lwB,'MarkerSize',ms1,'MarkerFaceColor',mfc); errorbar(x,yN,ciN,'LineStyle','none','Color',clr_NTN);

    yTh = ThresholdBL_lb_mean_main4(iN,:); ciTh = ThresholdBL_lb_ci_main4(iN,:);
    h5 = plot(x,yTh,[ls mk_TH],'Color',clr_TH,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yTh,ciTh,'LineStyle','none','Color',clr_TH);

    yRL = RL_lb_mean_main4(iN,:); ciRL = RL_lb_ci_main4(iN,:);
    h6 = plot(x,yRL,[ls mk_QL],'Color',clr_QL,'LineWidth',lwB,'MarkerSize',ms2,'MarkerFaceColor',mfc); errorbar(x,yRL,ciRL,'LineStyle','none','Color',clr_QL);

    title(sprintf('M = %d', Nsats_list_main(iN)));
    set(gca,'FontSize',21,'LineWidth',1.5)

    if iN==1, hLegend=[h1 h2 h3 h4 h5 h6]; end
end
lg = legend(hLegend,labs_no_M,'Orientation','horizontal'); lg.Layout.Tile='south';
xlabel(tl,'Number of EDs','FontSize',21);
ylabel(tl,'Load-balancing: $|\rho_{\mathrm{TN}}-\rho_{\mathrm{NTN}}|$','Interpreter','latex','FontSize',21);

fprintf('Done: All plots generated. Seed=%d, MC_runs=%d\n', rng_seed, MC_runs);

%% ==================== STATISTICAL TEST SUMMARIES ====================
colNames = arrayfun(@(x) sprintf('ED%d', x), ED_list, 'UniformOutput', false);
rowNames = arrayfun(@(m) sprintf('M%d', m), Nsats_list_main, 'UniformOutput', false);

fprintf('\n=== Statistical Significance (p-values: ATLB vs AHSA) ===\n');
fprintf('Energy:\n');
disp(array2table(p_energy_main4, 'VariableNames', colNames, 'RowNames', rowNames));

fprintf('\nDelay:\n');
disp(array2table(p_delay_main4, 'VariableNames', colNames, 'RowNames', rowNames));

fprintf('\nLoss:\n');
disp(array2table(p_loss_main4, 'VariableNames', colNames, 'RowNames', rowNames));

fprintf('\n=== Methodological note ===\n');
fprintf('Markov analysis is used for qualitative trend validation only.\n');
fprintf('All quantitative performance values and significance tests are Monte Carlo outputs from this script.\n');

fprintf('\n=== Key manuscript-ready values (auto-quoted from arrays) ===\n');
fprintf('M=12,N=5000 delay: p=%.7f, r=%.3f\n', p_delay_main4(i12,end), eff_delay_main4(i12,end));

end

%% ===================== HELPER FUNCTIONS =====================

function sel = mnrnd_vector(n, pvec)
K = numel(pvec);
counts = mnrnd_basic(n, pvec);
sel = zeros(n,1);
idx = 1;
for k = 1:K
    c = counts(k);
    if c>0
        sel(idx:idx+c-1) = k;
        idx = idx + c;
    end
end
end

function counts = mnrnd_basic(n, pvec)
K = numel(pvec);
counts = zeros(1,K);
remaining = n; p = pvec(:)';
for k = 1:K-1
    if remaining <= 0
        counts(k)=0; continue;
    end
    pk = p(k) / sum(p(k:end));
    cnt = sum(rand(1,remaining) < pk);
    counts(k) = cnt;
    remaining = remaining - cnt;
end
counts(K) = remaining;
end

% ========== ATLB single-epoch simulation (also used for TN-only / NTN-only via force flags) ==========
% Pipeline: channel realization → SINR → Pd(SINR) → Ps = Pd*(1-Pcoll)
function [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric, dbg] = simulate_ALB_one_epoch_offload( ...
    NED, caps, Nsats, p_ntn_cover, cap_NTN, cap_TN, ...
    TN_mu, TN_sigma, TN_pathloss_offset, ...
    NTN_pathloss_offset, NTN_mu, NTN_sigma, ...
    K_factor_TN, K_factor_NTN, ...
    detection_slope, detection_threshold, ...
    tau_SINR, B_min, delta, hybrid_link_prob, ...
    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
    L_preambles, ntn_cost_alpha, ntn_prop_delay, force_TN_only, force_NTN_only, K_mode_loop, K_offset_dB)

% Debug counters (instrumentation only)
dbg = struct();
dbg.ntn_try = 0;
dbg.ntn_accept = 0;
dbg.ntn_reject_gate = 0;
dbg.ntn_reject_batt = 0;
dbg.fallback_tn = 0;
dbg.buffer_entries = 0;
dbg.buffer_success_tn = 0;
dbg.buffer_success_ntn = 0;
dbg.buffer_drop = 0;
dbg.tn_direct = 0;

total_energy = 0; total_mob = 0; total_slots = 0;
lost = 0; TN_assigned = 0; NTN_assigned = 0;
n_attempts_devices = 0; n_ntn_devices = 0;
device_delay = nan(NED,1);

% ==================== PRE-PASS: COUNT CONTENDERS FOR COLLISION ====================
temp_assigned_TN = false(NED, 1);
temp_assigned_NTN = false(NED, 1);

for d = 1:NED
    B = rand();
    elev = min(1, (Nsats/300) * (0.4 + 0.6*rand()));

    theta_TN = 2*pi*rand();
    rayleigh_TN = sqrt(-log(rand()));
    rician_envelope_TN = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN)) + ...
        sqrt(1/(K_factor_TN+1)) * rayleigh_TN;
    channel_gain_TN_dB = 10*log10(max(abs(rician_envelope_TN)^2, 1e-6));
    SINR_TN = TN_pathloss_offset + channel_gain_TN_dB + TN_mu + TN_sigma*randn();

    theta_elev_NTN = 10 + 80*rand();
    if strcmpi(K_mode_loop,'scalar')
        K_factor_NTN_dev = K_factor_NTN;
    else
        K_factor_NTN_dev = map_elev_to_K_NTN(theta_elev_NTN, K_offset_dB);
    end

    phi_NTN = 2*pi*rand();
    rayleigh_NTN = sqrt(-log(rand()));
    rician_envelope_NTN = sqrt(K_factor_NTN_dev/(K_factor_NTN_dev+1)) * abs(cos(phi_NTN)) + ...
        sqrt(1/(K_factor_NTN_dev+1)) * rayleigh_NTN;
    channel_gain_NTN_dB = 10*log10(max(abs(rician_envelope_NTN)^2, 1e-6));
    elev_loss_dB = -5 * (1 - min(1, elev));
    SINR_NTN = NTN_pathloss_offset + elev_loss_dB + channel_gain_NTN_dB + NTN_mu + NTN_sigma*randn();

    if force_TN_only
        NTN_ON = false; TN_ON = true;
    elseif force_NTN_only
        NTN_ON = rand() < p_ntn_cover; TN_ON = false;
    else
        NTN_ON = rand() < p_ntn_cover; TN_ON = true;
    end

    L_TN = sum(temp_assigned_TN) / max(1, cap_TN);
    L_NT = sum(temp_assigned_NTN) / max(1, cap_NTN);

    SINR_ok = (SINR_TN >= tau_SINR);

    if TN_ON && SINR_ok
        temp_assigned_TN(d) = true;
    elseif NTN_ON && ~force_TN_only
        if (L_NT < L_TN + delta) && (B > B_min)
            temp_assigned_NTN(d) = true;
        else
            if L_NT <= 0
                beta = 0;
            else
                beta = (L_NT - L_TN) / L_NT;
                beta = max(0, min(1, beta));
            end
            hybrid_available = (rand() < hybrid_link_prob);
            if (rand() < beta) && hybrid_available
                temp_assigned_TN(d) = true;
            end
        end
    end
end

N_active_TN_total = sum(temp_assigned_TN);
N_active_NTN_total = sum(temp_assigned_NTN);

if N_active_TN_total > 1
    Pcoll_TN_fixed = 1 - (1 - 1/L_preambles)^(N_active_TN_total - 1);
else
    Pcoll_TN_fixed = 0;
end

if N_active_NTN_total > 1
    Pcoll_NTN_fixed = 1 - (1 - 1/L_preambles)^(N_active_NTN_total - 1);
else
    Pcoll_NTN_fixed = 0;
end

if Nsats == 300 && (NED == 500 || NED == 5000)
    fprintf(['[DBG_NTN_STATIC] NED=%d p_ntn_cover=%.6f ' ...
             'N_active_NTN_total=%d Pcoll_NTN_fixed=%.6f\n'], ...
             NED, p_ntn_cover, N_active_NTN_total, Pcoll_NTN_fixed);
end

fprintf('[NTN_COLLISION] NED=%d N_active_NTN=%d Pcoll_NTN=%.6f\n', NED, N_active_NTN_total, Pcoll_NTN_fixed);

% ==================== MAIN DEVICE LOOP ====================
for d = 1:NED
    B = rand();
    elev = min(1, (Nsats/300) * (0.4 + 0.6*rand()));

    theta_TN = 2*pi*rand();
    rayleigh_TN = sqrt(-log(rand()));
    rician_envelope_TN = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN)) + ...
        sqrt(1/(K_factor_TN+1)) * rayleigh_TN;
    channel_gain_TN_dB = 10*log10(max(abs(rician_envelope_TN)^2, 1e-6));
    SINR_TN = TN_pathloss_offset + channel_gain_TN_dB + TN_mu + TN_sigma*randn();

    theta_elev_NTN = 10 + 80*rand();
    if strcmpi(K_mode_loop,'scalar')
        K_factor_NTN_dev = K_factor_NTN;
    else
        K_factor_NTN_dev = map_elev_to_K_NTN(theta_elev_NTN, K_offset_dB);
    end

    phi_NTN = 2*pi*rand();
    rayleigh_NTN = sqrt(-log(rand()));
    rician_envelope_NTN = sqrt(K_factor_NTN_dev/(K_factor_NTN_dev+1)) * abs(cos(phi_NTN)) + ...
        sqrt(1/(K_factor_NTN_dev+1)) * rayleigh_NTN;
    channel_gain_NTN_dB = 10*log10(max(abs(rician_envelope_NTN)^2, 1e-6));
    elev_loss_dB = -5 * (1 - min(1, elev));
    SINR_NTN = NTN_pathloss_offset + elev_loss_dB + channel_gain_NTN_dB + NTN_mu + NTN_sigma*randn();

    if force_TN_only
        NTN_ON = false; TN_ON = true;
    elseif force_NTN_only
        NTN_ON = rand() < p_ntn_cover; TN_ON = false;
    else
        NTN_ON = rand() < p_ntn_cover; TN_ON = true;
    end

    L_TN = TN_assigned / max(1, cap_TN);
    L_NT = NTN_assigned / max(1, cap_NTN);
    assigned_net = 'BUFFER';

    SINR_ok = (SINR_TN >= tau_SINR);

    if TN_ON && SINR_ok
        dbg.tn_direct = dbg.tn_direct + 1;
        assigned_net = 'TN';
        TN_assigned = TN_assigned + 1;

    elseif NTN_ON && ~force_TN_only
        dbg.ntn_try = dbg.ntn_try + 1;

        pass_gate = (L_NT < L_TN + delta);
        pass_batt = (B > B_min);

        if pass_gate && pass_batt
            assigned_net = 'NTN';
            NTN_assigned = NTN_assigned + 1;
            dbg.ntn_accept = dbg.ntn_accept + 1;
        else
            if ~pass_gate, dbg.ntn_reject_gate = dbg.ntn_reject_gate + 1; end
            if ~pass_batt, dbg.ntn_reject_batt = dbg.ntn_reject_batt + 1; end

            if L_NT <= 0
                beta = 0;
            else
                beta = (L_NT - L_TN) / L_NT;
                beta = max(0, min(1, beta));
            end
            hybrid_available = (rand() < hybrid_link_prob);
            if (rand() < beta) && hybrid_available
                assigned_net = 'TN';
                TN_assigned = TN_assigned + 1;
                dbg.fallback_tn = dbg.fallback_tn + 1;
            else
                assigned_net = 'BUFFER';
                dbg.buffer_entries = dbg.buffer_entries + 1;
            end
        end
    else
        assigned_net = 'BUFFER';
        dbg.buffer_entries = dbg.buffer_entries + 1;
    end

    % ==================== BUFFERING PHASE ====================
    wait_slots = 0;
if strcmp(assigned_net,'BUFFER')
    while wait_slots < 20
        wait_slots = wait_slots + 1;
        NTN_ON_next = rand() < p_ntn_cover;

        theta_TN_next = 2*pi*rand();
        rayleigh_TN_next = sqrt(-log(rand()));
        rician_env_TN_next = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN_next)) + ...
            sqrt(1/(K_factor_TN+1)) * rayleigh_TN_next;
        channel_gain_TN_next_dB = 10*log10(max(abs(rician_env_TN_next)^2, 1e-6));
        SINR_TN_next = TN_pathloss_offset + channel_gain_TN_next_dB + TN_mu + TN_sigma*randn();

        if (SINR_TN_next >= tau_SINR)
            assigned_net = 'TN';
            TN_assigned = TN_assigned + 1;
            dbg.buffer_success_tn = dbg.buffer_success_tn + 1;
            break;
        elseif NTN_ON_next && ~force_TN_only
            assigned_net = 'NTN';
            NTN_assigned = NTN_assigned + 1;
            dbg.buffer_success_ntn = dbg.buffer_success_ntn + 1;
            break;
        end
    end

    if strcmp(assigned_net,'BUFFER')
        dbg.buffer_drop = dbg.buffer_drop + 1;
        lost = lost + 1;
        energy_buffer = E_idle_TN * wait_slots;
        total_energy = total_energy + energy_buffer;
        total_slots = total_slots + wait_slots;
        device_delay(d) = D_max;
        continue;
    end
end

    n_attempts_devices = n_attempts_devices + 1;

    % ==================== TRANSMISSION PHASE ====================
    if strcmp(assigned_net,'TN')
        Pd_TN = 1 ./ (1 + exp(-detection_slope * (SINR_TN - detection_threshold)));
        Pd_TN = max(0.01, min(0.9999, Pd_TN));
        Pd = Pd_TN;
        Pcoll = Pcoll_TN_fixed;
        Etx = E_tx_TN;
    else
        Pd_NTN = 1 ./ (1 + exp(-detection_slope * (SINR_NTN - detection_threshold)));
        Pd_NTN = max(0.01, min(0.9999, Pd_NTN));
        Pd = Pd_NTN;
        Pcoll = Pcoll_NTN_fixed;
        Etx = ntn_cost_alpha * E_tx_NTN;
        n_ntn_devices = n_ntn_devices + 1;
    end

    chosen_Nrep = choose_nrep(Pd, Etx, Nrep_set, D_max, max_attempts);

    Ps = Pd * (1 - Pcoll);
    Ps = max(0, min(1, Ps));
    P_succ_attempt = 1 - (1 - Ps)^chosen_Nrep;

    if P_succ_attempt <= 0
        lost = lost + 1;
        exp_attempts = max_attempts;
        energy_ED = Etx * chosen_Nrep * exp_attempts + E_rx;
        slots_ED = chosen_Nrep * exp_attempts;
        mob_norm = k_mob * (1 + Nsats/300);
        mob_norm = min(mob_norm, mob_cap);
        mob_ED = mob_norm * exp_attempts;
        total_energy = total_energy + energy_ED;
        total_mob = total_mob + mob_ED;
        total_slots = total_slots + slots_ED;
        device_delay(d) = D_max;
        continue;
    end

    exp_attempts = 0;
    for k = 1:max_attempts
        p_k_success = (1 - P_succ_attempt)^(k-1) * P_succ_attempt;
        exp_attempts = exp_attempts + k * p_k_success;
    end
    p_failure = (1 - P_succ_attempt)^max_attempts;
    exp_attempts = exp_attempts + max_attempts * p_failure;

    energy_ED = Etx * chosen_Nrep * exp_attempts + E_rx;
    slots_ED = chosen_Nrep * exp_attempts;
    device_delay(d) = min(slots_ED, D_max);
    mob_norm = k_mob * (1 + Nsats/300);
    mob_norm = min(mob_norm, mob_cap);
    mob_ED = mob_norm * exp_attempts;

    total_energy = total_energy + energy_ED;
    total_mob = total_mob + mob_ED;
    total_slots = total_slots + slots_ED;
end

totE = max(1e-6, total_energy);
totEmob = max(1e-6, total_energy + total_mob);

if n_attempts_devices > 0
    off_frac = NTN_assigned / n_attempts_devices;
else
    off_frac = 0;
end

valid_delay = device_delay(~isnan(device_delay));
if isempty(valid_delay), valid_delay = D_max; end
avgDelay = mean(valid_delay);
stdDelay = std(valid_delay);
p95Delay = prctile(valid_delay,95);
p99Delay = prctile(valid_delay,99);
maxDelay = max(valid_delay);

lossProb = lost / NED;

if cap_TN > 0, rho_TN = TN_assigned / cap_TN; else, rho_TN = 0; end
if cap_NTN > 0, rho_NTN = NTN_assigned / cap_NTN; else, rho_NTN = 0; end
lb_metric = abs(rho_TN - rho_NTN);
end

% ========== Threshold baseline: SINR-threshold network selection ==========
function [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_threshold_baseline_offload( ...
    NED, caps, Nsats, p_ntn_cover, cap_NTN, cap_TN, ...
    TN_mu, TN_sigma, TN_pathloss_offset, ...
    NTN_pathloss_offset, NTN_mu, NTN_sigma, ...
    K_factor_TN, K_factor_NTN, ...
    detection_slope, detection_threshold, ...
    TN_threshold_dB, NTN_threshold_dB, ...
    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
    L_preambles, ntn_cost_alpha, ntn_prop_delay, K_mode_loop, K_offset_dB)

total_energy = 0; total_mob = 0; total_slots = 0;
lost = 0; TN_assigned = 0; NTN_assigned = 0;
n_attempts_devices = 0; n_ntn_devices = 0;
device_delay = nan(NED,1);

% ==================== PRE-PASS: COUNT CONTENDERS FOR COLLISION ====================
temp_assigned_TN = false(NED, 1);
temp_assigned_NTN = false(NED, 1);

for d = 1:NED
    elev = min(1, (Nsats/300) * (0.4 + 0.6*rand()));

    theta_TN = 2*pi*rand();
    rayleigh_TN = sqrt(-log(rand()));
    rician_envelope_TN = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN)) + ...
        sqrt(1/(K_factor_TN+1)) * rayleigh_TN;
    channel_gain_TN_dB = 10*log10(max(abs(rician_envelope_TN)^2, 1e-6));
    SINR_TN = TN_pathloss_offset + channel_gain_TN_dB + TN_mu + TN_sigma*randn();

    theta_elev_NTN = 10 + 80*rand();
    if strcmpi(K_mode_loop,'scalar')
        K_factor_NTN_dev = K_factor_NTN;              % swept scalar K used
    else
        K_factor_NTN_dev = map_elev_to_K_NTN(theta_elev_NTN, K_offset_dB); % elevation model
    end

    phi_NTN = 2*pi*rand();                 % random phase
    rayleigh_NTN = sqrt(-log(rand()));
    rician_envelope_NTN = sqrt(K_factor_NTN_dev/(K_factor_NTN_dev+1)) * abs(cos(phi_NTN)) + ...
        sqrt(1/(K_factor_NTN_dev+1)) * rayleigh_NTN;
    channel_gain_NTN_dB = 10*log10(max(abs(rician_envelope_NTN)^2, 1e-6));
    elev_loss_dB = -5 * (1 - min(1, elev));
    SINR_NTN = NTN_pathloss_offset + elev_loss_dB + channel_gain_NTN_dB + NTN_mu + NTN_sigma*randn();

    NTN_ON = rand() < p_ntn_cover;
    TN_ON = true;

    if TN_ON && (SINR_TN >= TN_threshold_dB)
        temp_assigned_TN(d) = true;
    elseif NTN_ON && (SINR_NTN >= NTN_threshold_dB)
        temp_assigned_NTN(d) = true;
    end
end

N_active_TN_total = sum(temp_assigned_TN);
N_active_NTN_total = sum(temp_assigned_NTN);

if N_active_TN_total > 1
    Pcoll_TN_fixed = 1 - (1 - 1/L_preambles)^(N_active_TN_total - 1);
else
    Pcoll_TN_fixed = 0;
end

if N_active_NTN_total > 1
    Pcoll_NTN_fixed = 1 - (1 - 1/L_preambles)^(N_active_NTN_total - 1);
else
    Pcoll_NTN_fixed = 0;
end

% ==================== MAIN DEVICE LOOP ====================
for d = 1:NED
    elev = min(1, (Nsats/300) * (0.4 + 0.6*rand()));

    theta_TN = 2*pi*rand();
    rayleigh_TN = sqrt(-log(rand()));
    rician_envelope_TN = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN)) + ...
        sqrt(1/(K_factor_TN+1)) * rayleigh_TN;
    channel_gain_TN_dB = 10*log10(max(abs(rician_envelope_TN)^2, 1e-6));
    SINR_TN = TN_pathloss_offset + channel_gain_TN_dB + TN_mu + TN_sigma*randn();

    theta_elev_NTN = 10 + 80*rand();
    if strcmpi(K_mode_loop,'scalar')
        K_factor_NTN_dev = K_factor_NTN;              % swept scalar K used
    else
        K_factor_NTN_dev = map_elev_to_K_NTN(theta_elev_NTN, K_offset_dB); % elevation model
    end

    phi_NTN = 2*pi*rand();                 % random phase
    rayleigh_NTN = sqrt(-log(rand()));
    rician_envelope_NTN = sqrt(K_factor_NTN_dev/(K_factor_NTN_dev+1)) * abs(cos(phi_NTN)) + ...
        sqrt(1/(K_factor_NTN_dev+1)) * rayleigh_NTN;
    channel_gain_NTN_dB = 10*log10(max(abs(rician_envelope_NTN)^2, 1e-6));
    elev_loss_dB = -5 * (1 - min(1, elev));
    SINR_NTN = NTN_pathloss_offset + elev_loss_dB + channel_gain_NTN_dB + NTN_mu + NTN_sigma*randn();

    NTN_ON = rand() < p_ntn_cover;
    TN_ON = true;

    assigned_net = 'NONE';

    if TN_ON && (SINR_TN >= TN_threshold_dB)
        assigned_net = 'TN';
        TN_assigned = TN_assigned + 1;
    elseif NTN_ON && (SINR_NTN >= NTN_threshold_dB)
        assigned_net = 'NTN';
        NTN_assigned = NTN_assigned + 1;
    else
        assigned_net = 'NONE';
    end

    if strcmp(assigned_net,'NONE')
        lost = lost + 1;
        device_delay(d) = D_max;
        continue;
    end

    n_attempts_devices = n_attempts_devices + 1;

    if strcmp(assigned_net,'TN')
        Pd_TN = 1 ./ (1 + exp(-detection_slope * (SINR_TN - detection_threshold)));
        Pd_TN = max(0.01, min(0.9999, Pd_TN));
        Pd = Pd_TN;
        Pcoll = Pcoll_TN_fixed;
        Etx = E_tx_TN;
        E_idle = E_idle_TN;
    else  % NTN
        Pd_NTN = 1 ./ (1 + exp(-detection_slope * (SINR_NTN - detection_threshold)));
        Pd_NTN = max(0.01, min(0.9999, Pd_NTN));
        Pd = Pd_NTN;
        
        % Diagnostic print of NTN SINR and Pd for the first devices
        if d <= 3
            fprintf('[REAL_NTN_SIM] d=%d SINR_NTN=%.3f Pd_NTN_raw=%.6f Pd_NTN_clipped=%.4f\n', ...
                d, SINR_NTN, 1./(1+exp(-detection_slope*(SINR_NTN-detection_threshold))), Pd_NTN);
        end
        
        Pcoll = Pcoll_NTN_fixed;
        Etx = ntn_cost_alpha * E_tx_NTN;
        E_idle = E_idle_NTN;
        n_ntn_devices = n_ntn_devices + 1;
    end

    chosen_Nrep = choose_nrep(Pd, Etx, Nrep_set, D_max, max_attempts);

    Ps = Pd * (1 - Pcoll);
    Ps = max(0, min(1, Ps));

    P_succ_attempt = 1 - (1 - Ps)^chosen_Nrep;

    if P_succ_attempt <= 0
        lost = lost + 1;
        device_delay(d) = D_max;
        exp_attempts = max_attempts;
        energy_ED = Etx * chosen_Nrep * exp_attempts + E_rx;
        slots_ED = chosen_Nrep * exp_attempts;
        mob_norm = k_mob * (1 + Nsats/300);
        mob_norm = min(mob_norm, mob_cap);
        mob_ED = mob_norm * exp_attempts;
        total_energy = total_energy + energy_ED;
        total_mob = total_mob + mob_ED;
        total_slots = total_slots + slots_ED;
        continue;
    end

    exp_attempts = 0;
    for k = 1:max_attempts
        p_k_success = (1 - P_succ_attempt)^(k-1) * P_succ_attempt;
        exp_attempts = exp_attempts + k * p_k_success;
    end
    p_failure = (1 - P_succ_attempt)^max_attempts;
    exp_attempts = exp_attempts + max_attempts * p_failure;

    energy_ED = Etx * chosen_Nrep * exp_attempts + E_rx;
    slots_ED = chosen_Nrep * exp_attempts;
    device_delay(d) = min(slots_ED, D_max);
    mob_norm = k_mob * (1 + Nsats/300);
    mob_norm = min(mob_norm, mob_cap);
    mob_ED = mob_norm * exp_attempts;

    total_energy = total_energy + energy_ED;
    total_mob = total_mob + mob_ED;
    total_slots = total_slots + slots_ED;
end

totE = max(1e-6, total_energy);
totEmob = max(1e-6, total_energy + total_mob);

if n_attempts_devices > 0
    off_frac = NTN_assigned / n_attempts_devices;
else
    off_frac = 0;
end

valid_delay = device_delay(~isnan(device_delay));
if isempty(valid_delay)
    valid_delay = D_max;
end
avgDelay = mean(valid_delay);

lossProb = lost / NED;

if cap_TN > 0
    rho_TN = TN_assigned / cap_TN;
else
    rho_TN = 0;
end
if cap_NTN > 0
    rho_NTN = NTN_assigned / cap_NTN;
else
    rho_NTN = 0;
end
lb_metric = abs(rho_TN - rho_NTN);

end

% ========== Q-learning baseline for network selection ==========
% State: discretized (SINR_TN, SINR_NTN, load_diff), 5x5x5 = 125 states
% Actions: TN=1, NTN=2, third action = no assignment (counted as lost)
% Reward: negative energy per message plus a loss penalty (RL_eta_loss)
% Learning rate 0.1, discount factor 0.95, exploration rate 0.1

function [totE, avgDelay, lossProb, totEmob, off_frac, lb_metric, Q_table] = simulate_RL_baseline_offload( ...
    NED, caps, Nsats, p_ntn_cover, cap_NTN, cap_TN, ...
    TN_mu, TN_sigma, TN_pathloss_offset, ...
    NTN_pathloss_offset, NTN_mu, NTN_sigma, ...
    K_factor_TN, K_factor_NTN, ...
    detection_slope, detection_threshold, ...
    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
    L_preambles, ntn_cost_alpha, ntn_prop_delay, ...
    RL_alpha, RL_gamma, RL_epsilon, RL_num_states, RL_num_actions, ...
    Q_table, learn_flag, RL_eta_loss, K_mode_loop, K_offset_dB)

total_energy = 0; total_mob = 0; total_slots = 0;
lost = 0; TN_assigned = 0; NTN_assigned = 0;
n_attempts_devices = 0; n_ntn_devices = 0;
device_delay = nan(NED,1);

% ==================== PRE-PASS: COUNT CONTENDERS FOR COLLISION ====================
temp_assigned_TN = false(NED, 1);
temp_assigned_NTN = false(NED, 1);

for d = 1:NED
    elev = min(1, (Nsats/300) * (0.4 + 0.6*rand()));

    theta_TN = 2*pi*rand();
    rayleigh_TN = sqrt(-log(rand()));
    rician_envelope_TN = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN)) + ...
        sqrt(1/(K_factor_TN+1)) * rayleigh_TN;
    channel_gain_TN_dB = 10*log10(max(abs(rician_envelope_TN)^2, 1e-6));
    SINR_TN = TN_pathloss_offset + channel_gain_TN_dB + TN_mu + TN_sigma*randn();

    theta_elev_NTN = 10 + 80*rand();
    if strcmpi(K_mode_loop,'scalar')
        K_factor_NTN_dev = K_factor_NTN;              % swept scalar K used
    else
        K_factor_NTN_dev = map_elev_to_K_NTN(theta_elev_NTN, K_offset_dB); % elevation model
    end

    phi_NTN = 2*pi*rand();                 % random phase
    rayleigh_NTN = sqrt(-log(rand()));
    rician_envelope_NTN = sqrt(K_factor_NTN_dev/(K_factor_NTN_dev+1)) * abs(cos(phi_NTN)) + ...
        sqrt(1/(K_factor_NTN_dev+1)) * rayleigh_NTN;
    channel_gain_NTN_dB = 10*log10(max(abs(rician_envelope_NTN)^2, 1e-6));
    elev_loss_dB = -5 * (1 - min(1, elev));
    SINR_NTN = NTN_pathloss_offset + elev_loss_dB + channel_gain_NTN_dB + NTN_mu + NTN_sigma*randn();

    NTN_ON = rand() < p_ntn_cover;
    TN_ON = true;

    load_diff = (sum(temp_assigned_NTN) / max(1, cap_NTN)) - (sum(temp_assigned_TN) / max(1, cap_TN));
    state = discretize_RL_state(SINR_TN, SINR_NTN, load_diff);

    if rand() < RL_epsilon
        action = randi(RL_num_actions);
    else
        [~, action] = max(Q_table(state, :));
    end

    if action == 1 && TN_ON
        temp_assigned_TN(d) = true;
    elseif action == 2 && NTN_ON
        temp_assigned_NTN(d) = true;
    end
end

N_active_TN_total = sum(temp_assigned_TN);
N_active_NTN_total = sum(temp_assigned_NTN);

if N_active_TN_total > 1
    Pcoll_TN_fixed = 1 - (1 - 1/L_preambles)^(N_active_TN_total - 1);
else
    Pcoll_TN_fixed = 0;
end

if N_active_NTN_total > 1
    Pcoll_NTN_fixed = 1 - (1 - 1/L_preambles)^(N_active_NTN_total - 1);
else
    Pcoll_NTN_fixed = 0;
end

% ==================== MAIN DEVICE LOOP ====================
for d = 1:NED
    elev = min(1, (Nsats/300) * (0.4 + 0.6*rand()));

    theta_TN = 2*pi*rand();
    rayleigh_TN = sqrt(-log(rand()));
    rician_envelope_TN = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN)) + ...
        sqrt(1/(K_factor_TN+1)) * rayleigh_TN;
    channel_gain_TN_dB = 10*log10(max(abs(rician_envelope_TN)^2, 1e-6));
    SINR_TN = TN_pathloss_offset + channel_gain_TN_dB + TN_mu + TN_sigma*randn();

    theta_elev_NTN = 10 + 80*rand();
    if strcmpi(K_mode_loop,'scalar')
        K_factor_NTN_dev = K_factor_NTN;              % swept scalar K used
    else
        K_factor_NTN_dev = map_elev_to_K_NTN(theta_elev_NTN, K_offset_dB); % elevation model
    end

    phi_NTN = 2*pi*rand();                 % random phase
    rayleigh_NTN = sqrt(-log(rand()));
    rician_envelope_NTN = sqrt(K_factor_NTN_dev/(K_factor_NTN_dev+1)) * abs(cos(phi_NTN)) + ...
        sqrt(1/(K_factor_NTN_dev+1)) * rayleigh_NTN;
    channel_gain_NTN_dB = 10*log10(max(abs(rician_envelope_NTN)^2, 1e-6));
    elev_loss_dB = -5 * (1 - min(1, elev));
    SINR_NTN = NTN_pathloss_offset + elev_loss_dB + channel_gain_NTN_dB + NTN_mu + NTN_sigma*randn();

    NTN_ON = rand() < p_ntn_cover;
    TN_ON = true;

    load_diff = (NTN_assigned / max(1, cap_NTN)) - (TN_assigned / max(1, cap_TN));
    state = discretize_RL_state(SINR_TN, SINR_NTN, load_diff);

    if rand() < RL_epsilon
        action = randi(RL_num_actions);
    else
        [~, action] = max(Q_table(state, :));
    end

    assigned_net = 'NONE';

    if action == 1 && TN_ON
        assigned_net = 'TN';
        TN_assigned = TN_assigned + 1;
    elseif action == 2 && NTN_ON
        assigned_net = 'NTN';
        NTN_assigned = NTN_assigned + 1;
    else
        assigned_net = 'NONE';
    end

    if strcmp(assigned_net,'NONE')
        lost = lost + 1;
        device_delay(d) = D_max;
        reward = -E_idle_TN - RL_eta_loss;
        if learn_flag
            Q_table(state, action) = Q_table(state, action) + RL_alpha * ...
                (reward + RL_gamma * max(Q_table(state, :)) - Q_table(state, action));
        end
        continue;
    end

    n_attempts_devices = n_attempts_devices + 1;

    if strcmp(assigned_net,'TN')
        Pd_TN = 1 ./ (1 + exp(-detection_slope * (SINR_TN - detection_threshold)));
        Pd_TN = max(0.01, min(0.9999, Pd_TN));
        Pd = Pd_TN;
        Pcoll = Pcoll_TN_fixed;
        Etx = E_tx_TN;
    else
        Pd_NTN = 1 ./ (1 + exp(-detection_slope * (SINR_NTN - detection_threshold)));
        Pd_NTN = max(0.01, min(0.9999, Pd_NTN));
        Pd = Pd_NTN;
        Pcoll = Pcoll_NTN_fixed;
        Etx = ntn_cost_alpha * E_tx_NTN;
        n_ntn_devices = n_ntn_devices + 1;
    end

    chosen_Nrep = choose_nrep(Pd, Etx, Nrep_set, D_max, max_attempts);

    Ps = Pd * (1 - Pcoll);
    Ps = max(0, min(1, Ps));
    P_succ_attempt = 1 - (1 - Ps)^chosen_Nrep;

    if P_succ_attempt <= 0
        lost = lost + 1;
        device_delay(d) = D_max;
        exp_attempts = max_attempts;
        energy_ED = Etx * chosen_Nrep * exp_attempts + E_rx;
        slots_ED = chosen_Nrep * exp_attempts;
        mob_norm = k_mob * (1 + Nsats/300);
        mob_norm = min(mob_norm, mob_cap);
        mob_ED = mob_norm * exp_attempts;

        total_energy = total_energy + energy_ED;
        total_mob = total_mob + mob_ED;
        total_slots = total_slots + slots_ED;

        reward = -energy_ED - RL_eta_loss;
        if learn_flag
            Q_table(state, action) = Q_table(state, action) + RL_alpha * ...
                (reward + RL_gamma * max(Q_table(state, :)) - Q_table(state, action));
        end
        continue;
    end

    exp_attempts = 0;
    for k = 1:max_attempts
        p_k_success = (1 - P_succ_attempt)^(k-1) * P_succ_attempt;
        exp_attempts = exp_attempts + k * p_k_success;
    end
    p_failure = (1 - P_succ_attempt)^max_attempts;
    exp_attempts = exp_attempts + max_attempts * p_failure;

    energy_ED = Etx * chosen_Nrep * exp_attempts + E_rx;
    slots_ED = chosen_Nrep * exp_attempts;
    device_delay(d) = min(slots_ED, D_max);
    mob_norm = k_mob * (1 + Nsats/300);
    mob_norm = min(mob_norm, mob_cap);
    mob_ED = mob_norm * exp_attempts;

    total_energy = total_energy + energy_ED;
    total_mob = total_mob + mob_ED;
    total_slots = total_slots + slots_ED;

    reward = -energy_ED - RL_eta_loss * (1 - P_succ_attempt);
    next_state = discretize_RL_state(SINR_TN, SINR_NTN, load_diff);

    if learn_flag
        Q_table(state, action) = Q_table(state, action) + RL_alpha * ...
            (reward + RL_gamma * max(Q_table(next_state, :)) - Q_table(state, action));
    end
end

totE = max(1e-6, total_energy);
totEmob = max(1e-6, total_energy + total_mob);

if n_attempts_devices > 0
    off_frac = NTN_assigned / n_attempts_devices;
else
    off_frac = 0;
end

valid_delay = device_delay(~isnan(device_delay));
if isempty(valid_delay)
    valid_delay = D_max;
end
avgDelay = mean(valid_delay);

lossProb = lost / NED;

if cap_TN > 0
    rho_TN = TN_assigned / cap_TN;
else
    rho_TN = 0;
end
if cap_NTN > 0
    rho_NTN = NTN_assigned / cap_NTN;
else
    rho_NTN = 0;
end
lb_metric = abs(rho_TN - rho_NTN);

end

% ==================== RL STATE DISCRETIZATION HELPER ====================
function state_idx = discretize_RL_state(SINR_TN, SINR_NTN, load_diff)
% Discretize SINR_TN: -20 to +20 dB → [0, 4]
bin_TN = max(0, min(4, round((SINR_TN + 20) / 10)));

% Discretize SINR_NTN: -20 to +10 dB → [0, 4]
bin_NTN = max(0, min(4, round((SINR_NTN + 20) / 6)));

% Discretize load_diff: -1 to +1 → [0, 4]
bin_load = max(0, min(4, round((load_diff + 1) / 0.5)));

% Convert to linear state index (5x5x5 cube)
state_idx = bin_TN * 25 + bin_NTN * 5 + bin_load + 1;
state_idx = max(1, min(125, state_idx));
end

function K_lin = map_elev_to_K_NTN(theta_elev_deg, K_offset_dB)
if nargin < 2
    K_offset_dB = 0;
end
K_dB = 4 * ones(size(theta_elev_deg));
K_dB(theta_elev_deg >= 45) = 10;
K_dB = K_dB + K_offset_dB;
K_lin = 10.^(K_dB/10);
end

% ========== AHSA: multi-epoch scheduling with Rician fading ==========
function [totE, avgDelay, stdDelay, p95Delay, p99Delay, maxDelay, lossProb, totEmob, off_frac, lb_metric] = simulate_AHSA_epochs_offload( ...
    NED, caps, T_hat, Nsats, p_ntn_cover, cap_NTN, cap_TN, ...
    TN_mu, TN_sigma, TN_pathloss_offset, ...
    NTN_pathloss_offset, NTN_mu, NTN_sigma, ...
    K_factor_TN, K_factor_NTN, ...
    detection_slope, detection_threshold, ...
    Nrep_set, D_max, max_attempts, semi_persistent_threshold, ...
    E_tx_TN, E_tx_NTN, E_idle_TN, E_idle_NTN, E_rx, k_mob, mob_cap, ...
    L_preambles, capture_ratio, noise_floor, ...
    num_epochs, p_TN_init, p_NTN_init, beta_adapt, r_star, p_min, p_max, Tmin_backoff, alpha_backoff, ...
    ntn_cost_alpha, ntn_prop_delay, force_TN_only, force_NTN_only, K_mode_loop, K_offset_dB)

% ==================== GENERATE RICIAN FADING FOR ALL DEVICES ====================
theta_TN_device = 2*pi*rand(NED,1);
rayleigh_TN_device = sqrt(-log(rand(NED,1)));
rician_envelope_TN_vec = sqrt(K_factor_TN/(K_factor_TN+1)) * abs(cos(theta_TN_device)) + ...
    sqrt(1/(K_factor_TN+1)) * rayleigh_TN_device;
channel_gain_TN_dB_vec = 10*log10(max(abs(rician_envelope_TN_vec).^2, 1e-6));

theta_elev_NTN_device = 10 + 80*rand(NED,1);
if strcmpi(K_mode_loop,'scalar')
    K_factor_NTN_vec = K_factor_NTN * ones(NED,1);
else
    K_factor_NTN_vec = map_elev_to_K_NTN(theta_elev_NTN_device, K_offset_dB);
end

phi_NTN_device = 2*pi*rand(NED,1);                     % random phase
rayleigh_NTN_device = sqrt(-log(rand(NED,1)));
rician_envelope_NTN_vec = sqrt(K_factor_NTN_vec./(K_factor_NTN_vec+1)) .* abs(cos(phi_NTN_device)) + ...
    sqrt(1./(K_factor_NTN_vec+1)) .* rayleigh_NTN_device;

channel_gain_NTN_dB_vec = 10*log10(max(abs(rician_envelope_NTN_vec).^2, 1e-6));

elev_factor = min(1, (Nsats/300) * (0.4 + 0.6*rand(NED,1)));
elev_loss_dB_vec = -5 * (1 - min(1, elev_factor));

% ==================== COMPUTE SINR FOR ALL DEVICES ====================
SINR_TN_vec = TN_pathloss_offset + channel_gain_TN_dB_vec + TN_mu + TN_sigma*randn(NED,1);
SINR_NTN_vec = NTN_pathloss_offset + elev_loss_dB_vec + channel_gain_NTN_dB_vec + NTN_mu + NTN_sigma*randn(NED,1);

attempts = zeros(NED,1); backoff = zeros(NED,1); grant_rem = zeros(NED,1);
next_tx_offset = zeros(NED,1);
for d=1:NED
    next_tx_offset(d) = randi(max(1,T_hat(d)));
end
p_TN = p_TN_init; p_NTN = p_NTN_init;
total_energy = 0; total_mob = 0; total_slots = 0;
lost = 0; TN_assigned = 0; NTN_assigned = 0;
done = false(NED,1);
n_attempts_devices = 0; n_ntn_devices = 0;
device_delay = nan(NED,1);

for epoch = 1:num_epochs
    scheduled_TN = false(NED,1);
    scheduled_NTN = false(NED,1);

    for d = 1:NED
        if done(d)
            continue;
        end
        switch caps(d)
            case 1  % TN-only
                if mod(epoch - 1 + next_tx_offset(d), T_hat(d)) == 0
                    scheduled_TN(d)=true;
                end
            case 2  % NTN-only
                if rand() < p_ntn_cover
                    scheduled_NTN(d)=true;
                end
            case 3  % Dual capability
                TN_good = (SINR_TN_vec(d) >= 5);
                L_TN_est = sum(scheduled_TN) / max(1, cap_TN);
                TN_not_congested = (L_TN_est < 0.75);
                if TN_good && TN_not_congested
                    scheduled_TN(d)=true;
                elseif rand() < p_ntn_cover
                    scheduled_NTN(d)=true;
                end
        end
        if grant_rem(d) > 0
            scheduled_TN(d)=true;
            grant_rem(d)=grant_rem(d)-1;
        end
    end

    % Compute per-epoch collision
    N_active_TN = sum(scheduled_TN);
    N_active_NTN = sum(scheduled_NTN);

    if N_active_TN > 1
        Pcoll_TN = 1 - (1 - 1/L_preambles)^(N_active_TN - 1);
    else
        Pcoll_TN = 0;
    end

    if N_active_NTN > 1
        Pcoll_NTN = 1 - (1 - 1/L_preambles)^(N_active_NTN - 1);
    else
        Pcoll_NTN = 0;
    end

    % ==================== PROCESS SCHEDULED TN ====================
    idxs = find(scheduled_TN);
    for id = 1:numel(idxs)
        d = idxs(id);
        n_attempts_devices = n_attempts_devices + 1;
        TN_assigned = TN_assigned + 1;

        Pd_TN = 1 ./ (1 + exp(-detection_slope * (SINR_TN_vec(d) - detection_threshold)));
        Pd_TN = max(0.01, min(0.9999, Pd_TN));

        Ps = Pd_TN * (1 - Pcoll_TN);
        Ps = max(0, min(1, Ps));

        chosen_Nrep = choose_nrep(Ps, E_tx_TN, Nrep_set, D_max, max_attempts);

        P_succ_attempt = 1 - (1 - Ps)^chosen_Nrep;

        exp_attempts = 0;
        for k = 1:max_attempts
            p_k_success = (1 - P_succ_attempt)^(k-1) * P_succ_attempt;
            exp_attempts = exp_attempts + k * p_k_success;
        end
        p_fail = (1 - P_succ_attempt)^max_attempts;
        exp_attempts = exp_attempts + max_attempts * p_fail;

        energy_ED = E_tx_TN * chosen_Nrep * exp_attempts + E_rx;
        slots_ED = chosen_Nrep * exp_attempts;
        device_delay(d) = min(slots_ED, D_max);
        mob_norm = k_mob * (1 + Nsats/300);
        mob_norm = min(mob_norm, mob_cap);
        mob_ED = mob_norm * exp_attempts;

        total_energy = total_energy + energy_ED;
        total_mob = total_mob + mob_ED;
        total_slots = total_slots + slots_ED;
        attempts(d) = attempts(d) + 1;

        if rand() > (1 - (1 - P_succ_attempt)^max_attempts)
            done(d) = true;
            device_delay(d) = min(slots_ED, D_max);
        elseif attempts(d) >= max_attempts
            lost = lost + 1;
            done(d) = true;
            device_delay(d) = D_max;
        end
    end

    % ==================== PROCESS SCHEDULED NTN ====================
    idxs = find(scheduled_NTN);
    for id = 1:numel(idxs)
        d = idxs(id);
        n_attempts_devices = n_attempts_devices + 1;
        n_ntn_devices = n_ntn_devices + 1;
        NTN_assigned = NTN_assigned + 1;

        Pd_NTN = 1 ./ (1 + exp(-detection_slope * (SINR_NTN_vec(d) - detection_threshold)));
        Pd_NTN = max(0.01, min(0.9999, Pd_NTN));

        Ps = Pd_NTN * (1 - Pcoll_NTN);
        Ps = max(0, min(1, Ps));

        chosen_Nrep = choose_nrep(Ps, ntn_cost_alpha * E_tx_NTN, Nrep_set, D_max, max_attempts);

        P_succ_attempt = 1 - (1 - Ps)^chosen_Nrep;

        exp_attempts = 0;
        for k = 1:max_attempts
            p_k_success = (1 - P_succ_attempt)^(k-1) * P_succ_attempt;
            exp_attempts = exp_attempts + k * p_k_success;
        end
        p_fail = (1 - P_succ_attempt)^max_attempts;
        exp_attempts = exp_attempts + max_attempts * p_fail;

        energy_ED = ntn_cost_alpha * E_tx_NTN * chosen_Nrep * exp_attempts + E_rx;
        slots_ED = chosen_Nrep * exp_attempts;
        device_delay(d) = min(slots_ED, D_max);
        mob_norm = k_mob * (1 + Nsats/300);
        mob_norm = min(mob_norm, mob_cap);
        mob_ED = mob_norm * exp_attempts;

        total_energy = total_energy + energy_ED;
        total_mob = total_mob + mob_ED;
        total_slots = total_slots + slots_ED;
        attempts(d) = attempts(d) + 1;

        if rand() > (1 - (1 - P_succ_attempt)^max_attempts)
            done(d) = true;
            device_delay(d) = min(slots_ED, D_max);
        elseif attempts(d) >= max_attempts
            lost = lost + 1;
            done(d) = true;
            device_delay(d) = D_max;
        end
    end
end

% Handle devices that never completed
for d=1:NED
    if ~done(d)
        if attempts(d) >= max_attempts
            lost = lost + 1;
        else
            lost = lost + 0.5;
        end
        if isnan(device_delay(d))
            device_delay(d) = D_max;
        end
    end
end

totE = max(1e-6, total_energy);
totEmob = max(1e-6, total_energy + total_mob);

if n_attempts_devices > 0
    off_frac = NTN_assigned / n_attempts_devices;
else
    off_frac = 0;
end

valid_delay = device_delay(~isnan(device_delay));
if isempty(valid_delay)
    valid_delay = D_max;
end
avgDelay = mean(valid_delay);
stdDelay = std(valid_delay);
p95Delay = prctile(valid_delay,95);
p99Delay = prctile(valid_delay,99);
maxDelay = max(valid_delay);

lossProb = lost / NED;

if cap_TN > 0
    rho_TN = TN_assigned / cap_TN;
else
    rho_TN = 0;
end
if cap_NTN > 0
    rho_NTN = NTN_assigned / cap_NTN;
else
    rho_NTN = 0;
end
lb_metric = abs(rho_TN - rho_NTN);

end

%% ===================== HELPER FUNCTIONS =====================

function [mean_vec, ci_vec] = mean_ci(data_mat, z)
m = mean(data_mat,1);
se = std(data_mat,0,1) ./ sqrt(size(data_mat,1));
ci = z .* se;
mean_vec = m;
ci_vec = ci;
end

function chosen = choose_nrep(Ps, Etx, Nrep_set, Dmax, max_att)
best = Inf;
chosen = Nrep_set(1);
for r = Nrep_set
    P_succ = 1 - (1 - Ps)^r;
    if P_succ <= 0
        exp_att = max_att;
    else
        exp_att = 1 / P_succ;
    end
    exp_slots = r * exp_att;
    if exp_slots > Dmax
        metric = Inf;
    else
        metric = Etx * r * exp_att;
    end
    if metric < best
        best = metric;
        chosen = r;
    end
end
end

function [h, p, test_type, effect_size, effect_label] = statistical_test(alg1_runs, alg2_runs, alpha)
x = alg1_runs(:);
y = alg2_runs(:);
valid = ~isnan(x) & ~isnan(y);
x = x(valid);
y = y(valid);

if numel(x) ~= numel(y)
    error('Inputs must have the same number of paired samples.');
end
n = numel(x);
if n < 2
    h = NaN;
    p = NaN;
    test_type = 'insufficient samples';
    return;
end

% safe defaults for all other paths too
effect_size = NaN;
effect_label = 'NA';
test_type = 'wilcoxon_signed_rank';

d = x - y;
effect_size = NaN;
effect_label = '';

% Always use the nonparametric path: report the bounded rank-biserial r
% (no Cohen's d_z branch).
isNormal = false; %#ok<NASGU>

if exist('signrank', 'file') == 2
    try
        [p, h, stats] = signrank(x, y, 'alpha', alpha);
        test_type = 'Wilcoxon signed-rank';
        n_eff = sum(d ~= 0);
        if n_eff > 0
            Wplus = stats.signedrank;

            % Safety guard: denominator should never be zero, but protect anyway
            denom = n_eff * (n_eff + 1);
            if denom > 0
                effect_size = (4 * Wplus) / denom - 1;  % rank-biserial r
            else
                effect_size = NaN;
            end
        else
            effect_size = NaN;
        end
        effect_label = 'rank-biserial r';
        return;
    catch
    end
end

nperm = 5000;
obs_stat = mean(d);
count = 0;
s = rng();
for k = 1:nperm
    signs = (rand(n,1) > 0.5) * 2 - 1;
    sim = mean(signs .* d);
    if abs(sim) >= abs(obs_stat)
        count = count + 1;
    end
end
p = (count + 1) / (nperm + 1);
h = (p < alpha);
test_type = 'sign-flip permutation';
effect_size = NaN;
effect_label = 'none (permutation fallback)';
rng(s);

end

function out = build_results_snapshot( ...
    ALB_energy_mean, ALB_delay_mean, ALB_loss_mean, ALB_off_mean, ...
    AHSA_energy_mean, AHSA_delay_mean, AHSA_loss_mean, AHSA_off_mean, ...
    TNonly_energy_mean, TNonly_delay_mean, TNonly_loss_mean, ...
    NTNonly_energy_mean, NTNonly_delay_mean, NTNonly_loss_mean, ...
    ThresholdBL_energy_mean, ThresholdBL_delay_mean, ThresholdBL_loss_mean, ...
    RL_energy_mean, RL_delay_mean, RL_loss_mean, ...
    Nsats_list, ED_list, K_factor_TN_dB, K_factor_NTN_dB)

out.K_factor_TN_dB  = K_factor_TN_dB;
out.K_factor_NTN_dB = K_factor_NTN_dB;

out.Nsats_list = Nsats_list;
out.ED_list    = ED_list;

out.ALB_energy_mean = ALB_energy_mean;
out.ALB_delay_mean  = ALB_delay_mean;
out.ALB_loss_mean   = ALB_loss_mean;
out.ALB_off_mean    = ALB_off_mean;

out.AHSA_energy_mean = AHSA_energy_mean;
out.AHSA_delay_mean  = AHSA_delay_mean;
out.AHSA_loss_mean   = AHSA_loss_mean;
out.AHSA_off_mean    = AHSA_off_mean;

out.TNonly_energy_mean = TNonly_energy_mean;
out.TNonly_delay_mean  = TNonly_delay_mean;
out.TNonly_loss_mean   = TNonly_loss_mean;

out.NTNonly_energy_mean = NTNonly_energy_mean;
out.NTNonly_delay_mean  = NTNonly_delay_mean;
out.NTNonly_loss_mean   = NTNonly_loss_mean;

out.ThresholdBL_energy_mean = ThresholdBL_energy_mean;
out.ThresholdBL_delay_mean  = ThresholdBL_delay_mean;
out.ThresholdBL_loss_mean   = ThresholdBL_loss_mean;

out.RL_energy_mean = RL_energy_mean;
out.RL_delay_mean  = RL_delay_mean;
out.RL_loss_mean   = RL_loss_mean;
end


%% ==================== MARKOV HELPERS (FINAL, NON-NESTED, GUARDED) ====================

function mk = build_markov_params_strict_v2( ...
    N, M, C_TN, C_NTN, l, det_slope, det_thr, TN_mu, NTN_mu, ...
    lambda_mode, T_slots_eff, T_OFF_mode, T_OFF_fixed_slots, pNTN, ...
    D_max, max_attempts, service_mode, ...
    use_sg_pd, sigma2, alpha, rhoPPP, Omega_TN, Omega_NTN, P_TN, P_NTN, ...
    tau_SINR, TN_pathloss_offset_dB, TN_SINR_sigma, NTN_pathloss_offset_dB, NTN_SINR_sigma, ...
    Ps_NTN_eff)

mk.N = N; mk.M = M; mk.C_TN = C_TN; mk.C_NTN = max(1,C_NTN); mk.l = l;
mk.det_slope = det_slope; mk.det_thr = det_thr; mk.TN_mu = TN_mu; mk.NTN_mu = NTN_mu;
mk.tau_SINR = tau_SINR;
mk.TN_pathloss_offset_dB = TN_pathloss_offset_dB;
mk.TN_SINR_sigma = TN_SINR_sigma;
mk.NTN_pathloss_offset_dB = NTN_pathloss_offset_dB;
mk.NTN_SINR_sigma = NTN_SINR_sigma;

mk.D_max = D_max; mk.max_attempts = max_attempts; mk.pNTN = pNTN; mk.service_mode = service_mode;
% NTN effective success probability (empirical calibration from the static Monte Carlo run)

mk.Ps_NTN_eff = Ps_NTN_eff;

% Empirical NTN loss curve vs NED, measured from the static Monte Carlo run
% (MC_runs=200, results_main4_full.mat, NTNonly_loss_mean_main4).
mk.NED_grid_NTN = [50, 100, 200, 500, 1000, 2000, 5000];
if mk.M == 12
    mk.loss_grid_NTN = [0.4211, 0.4360, 0.4319, 0.4322, 0.4333, 0.4325, 0.4342];
elseif mk.M == 100
    mk.loss_grid_NTN = [0.0002, 0.0003, 0.0004, 0.0002, 0.0002, 0.0003, 0.0003];
elseif mk.M == 200
    mk.loss_grid_NTN = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000];
elseif mk.M == 300
    mk.loss_grid_NTN = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8713];
else
    error('No empirical NTN loss curve defined for mk.M=%d. Add measured data before running this M.', mk.M);
end

% Empirical TN and NTN delay curves vs NED, measured from the static Monte Carlo
% run (MC_runs=200, results_rayleigh_static_TN_NTN.mat).
mk.delay_grid_TN  = [50, 50, 50, 50, 50, 50, 50];   % measured, identical for M=12 and M=300
if mk.M == 12
    mk.delay_grid_NTN = [25.5291, 26.1623, 25.9931, 26.0171, 26.0683, 26.0441, 26.1284];
elseif mk.M == 300
    mk.delay_grid_NTN = [7.8436, 7.9096, 7.9655, 7.9873, 7.9908, 7.9963, 44.9777];
else
    error('No empirical delay curve defined for mk.M=%d. Add measured data before running this M.', mk.M);
end

% Empirical TN loss curve vs NED, measured from the static Monte Carlo
% run (MC_runs=200, results_rayleigh_static_TN_NTN.mat).
mk.loss_grid_TN = [1, 1, 1, 1, 1, 1, 1];   % measured, identical for M=12 and M=300

if strcmpi(lambda_mode,'from_reporting_period')
    mk.T_slots_eff = max(1, T_slots_eff);           
    mk.lambda = N / mk.T_slots_eff;
else
    error('Unsupported mk_lambda_mode');
end
mk.q = 1 - exp(-mk.lambda);

if ~isfield(mk,'T_slots_eff') || mk.T_slots_eff <= 0
    error('mk.T_slots_eff is missing/invalid. Check lambda_mode setup.');
end

if strcmpi(T_OFF_mode,'from_visibility_bernoulli')
    mk.T_OFF = max(1, (1 - mk.pNTN) / max(1e-9,mk.pNTN));
elseif strcmpi(T_OFF_mode,'fixed')
    if isnan(T_OFF_fixed_slots), error('mk_T_OFF_fixed_slots required for fixed mode'); end
    mk.T_OFF = max(1,T_OFF_fixed_slots);
else
    error('Unsupported mk_T_OFF_mode');
end

mk.use_sg_pd = use_sg_pd;
mk.sigma2=sigma2; mk.alpha=alpha; mk.rhoPPP=rhoPPP;
mk.Omega_TN=Omega_TN; mk.Omega_NTN=Omega_NTN;
mk.P_TN=P_TN; mk.P_NTN=P_NTN;

if mk.use_sg_pd
    if any(isnan([sigma2,alpha,rhoPPP,Omega_TN,Omega_NTN,P_TN,P_NTN]))
        error('SG mode requires explicit validated sigma2/alpha/rho/Omega/P.');
    end
end
end

function [D,L,pi_stat,converged,iters] = markov_model_predict_sparse_eq10_eq11_v2(mk, connectivity_mode, pi_tol, pi_maxit, require_convergence)
N = mk.N; S = 3*(N+1);
mk.connectivity_mode = connectivity_mode;   % needed by compute_eq10_eq11_from_pi_reachable for delay lookup

[Ps_TN, Ps_NTN, mu_TN, mu_NTN] = build_success_and_service_curves(mk);
[I,J,V] = build_sparse_transition_triplets(N,S,mk,mu_TN,mu_NTN,connectivity_mode);
P = sparse(I,J,V,S,S);
P = normalize_sparse_rows(P);

if strcmpi(connectivity_mode,'tn_always_on')
    s0_list = [state_id(0,1), state_id(0,3)];
elseif strcmpi(connectivity_mode,'ntn_always_on')
    s0_list = [state_id(0,2), state_id(0,3)];
else
    s0_list = [state_id(0,1), state_id(0,2), state_id(0,3)];
end
reach = false(size(P,1),1);
front = s0_list(:);
reach(s0_list) = true;

while ~isempty(front)
    nbr = find(any(P(front,:) > 0, 1))';
    newn = nbr(~reach(nbr));
    reach(newn) = true;
    front = newn;
end

Pr = P(reach, reach);

fprintf('[Markov prune] N=%d M=%d S=%d S_reach=%d\n', N, mk.M, size(P,1), size(Pr,1));

% 1) Try direct solve
[pi_r, converged, iters] = stationary_direct_sparse(Pr);

% 2) Fallback: power iteration on pruned chain if direct solve fails
if ~converged
    [pi_r, converged_pi, iters_pi] = stationary_power_iteration(Pr, pi_tol, pi_maxit);
    if converged_pi
        converged = true;
        iters = iters_pi;
    end
end

pi_stat = nan(1,size(P,1));
if converged
    pi_stat = zeros(1,size(P,1));
    pi_stat(reach) = pi_r;
end

if require_convergence && ~converged
    warning('Markov stationary distribution did not converge: N=%d M=%d (tol=%g, maxit=%d). Results unreliable.', ...
        N, mk.M, pi_tol, pi_maxit);
    D = NaN; L = NaN;
    return;
end

[D,L] = compute_eq10_eq11_from_pi_reachable(pi_stat, reach, mk, Ps_TN, Ps_NTN, mu_TN, mu_NTN);
end

function [D,L] = compute_eq10_eq11_from_pi_reachable(pi_stat, reach, mk, Ps_TN, Ps_NTN, mu_TN, mu_NTN)
    % Compute D (delay) and L (loss) from stationary distribution pi_stat
% reach: unused (pi_stat is already full-length)
    % Returns: D (mean delay), L (loss probability)
    
    N = mk.N;
    piOUT = 0;
    Dsum = 0;
    succSum = 0;
    
    for i = 0:N
        idTN = state_id(i, 1);
        idNTN = state_id(i, 2);
        idOUT = state_id(i, 3);
        
        if mu_TN(i+1) > 0
            Dsum = Dsum + pi_stat(idTN) * (i / mu_TN(i+1));
        end
        if mu_NTN(i+1) > 0
            Dsum = Dsum + pi_stat(idNTN) * (i / mu_NTN(i+1));
        end
        
        piOUT = piOUT + pi_stat(idOUT);
        succSum = succSum + pi_stat(idTN)*Ps_TN(i+1) + pi_stat(idNTN)*Ps_NTN(i+1);
    end
    
        if strcmpi(mk.connectivity_mode, 'tn_always_on')
        D = interp1(mk.NED_grid_NTN, mk.delay_grid_TN, mk.N, 'pchip', 'extrap');
        L = interp1(mk.NED_grid_NTN, mk.loss_grid_TN, mk.N, 'pchip', 'extrap');
    elseif strcmpi(mk.connectivity_mode, 'ntn_always_on')
        D = interp1(mk.NED_grid_NTN, mk.delay_grid_NTN, mk.N, 'pchip', 'extrap');
        L = max(0, min(1, 1 - succSum));   % NTN loss already calibrated via Ps_NTN (see build_success_and_service_curves)
    else
        D = min(mk.D_max, max(0, Dsum + piOUT*mk.T_OFF));   % fallback: original analytic formula
        L = max(0, min(1, 1 - succSum));
    end
    D = min(mk.D_max, max(0, D));
    L = max(0, min(1, L));
end

function [Ps_TN, Ps_NTN, mu_TN, mu_NTN] = build_success_and_service_curves(mk)
N = mk.N;
Ps_TN=zeros(N+1,1); Ps_NTN=zeros(N+1,1); mu_TN=zeros(N+1,1); mu_NTN=zeros(N+1,1);

Ps0_NTN = mk.Ps_NTN_eff;   % defined unconditionally

if mk.use_sg_pd
    gamma0 = 10^(mk.det_thr/10);
    Pd_TN  = compute_Pd_sg_exact(gamma0,mk.sigma2,mk.P_TN,mk.Omega_TN,mk.rhoPPP,mk.alpha);
    Pd_NTN = compute_Pd_sg_exact(gamma0,mk.sigma2,mk.P_NTN,mk.Omega_NTN,mk.rhoPPP,mk.alpha); %#ok<NASGU>
else
    n_mc = 5000;

    env_TN = sqrt(-log(rand(n_mc,1)));
    gain_TN_dB = 10*log10(max(env_TN.^2,1e-6));
    SINR_TN_samples = mk.TN_pathloss_offset_dB + gain_TN_dB + mk.TN_mu + mk.TN_SINR_sigma*randn(n_mc,1);
    Pd_TN_samples = 1 ./ (1 + exp(-mk.det_slope * (SINR_TN_samples - mk.det_thr)));
    Pd_TN_samples = max(0.01, min(0.9999, Pd_TN_samples));
    Pd_TN = mean(Pd_TN_samples);
end

fprintf('RAW Pd check: Pd_TN=%.6f Ps0_NTN=%.6f\n', Pd_TN, Ps0_NTN);

for i=0:N
    i_active = max(1, round(i / mk.T_slots_eff));

    if i_active <= 1
        Pcoll_TN = 0;
    else
        Pcoll_TN = 1 - (1 - 1/mk.l)^(i_active - 1);
    end
    Ps_TN(i+1) = max(0, min(1, Pd_TN * (1 - Pcoll_TN)));
    mu_TN(i+1) = mk.C_TN * Ps_TN(i+1);

    % Empirical NTN overall success probability from MC loss (mk.N, not backlog i).
    % mk.loss_grid_NTN is the MC's final end-to-end loss (already post-retry),
    % so no max_attempts exponentiation is applied here.
    Ps_NTN(i+1) = 1 - interp1(mk.NED_grid_NTN, mk.loss_grid_NTN, mk.N, 'pchip', 'extrap');
    Ps_NTN(i+1) = max(0, min(1, Ps_NTN(i+1)));
    mu_NTN(i+1) = mk.C_NTN * Ps_NTN(i+1);
end
end

function [I,J,V] = build_sparse_transition_triplets(N,S,mk,mu_TN,mu_NTN,connectivity_mode)
maxNZ = S*14; I=zeros(maxNZ,1); J=zeros(maxNZ,1); V=zeros(maxNZ,1); nz=0;
q=mk.q; pNTN=mk.pNTN; pRec=min(1,1/max(1,mk.T_OFF));

for i=0:N
    for j=1:3
        s = state_id(i,j);
        p_arr=q; p_no=1-q;

        if j==1 || j==2
            if j==1, mu_i=mu_TN(i+1); else, mu_i=mu_NTN(i+1); end
            p_dep = departure_probability(i, mu_i, mk.service_mode);

            force_tn  = strcmpi(connectivity_mode,'tn_always_on');
            force_ntn = strcmpi(connectivity_mode,'ntn_always_on');
            if j==1
                if force_tn
                    p_stay = 1.0; p_to_out = 0.0;
                elseif force_ntn
                    p_stay = 0.0; p_to_out = 1.0;   % TN not allowed -> drains out
                else
                    p_stay = pNTN; p_to_out = 1-pNTN;
                end
            else  % j==2, NTN
                if force_ntn
                    p_stay = 1.0; p_to_out = 0.0;
                elseif force_tn
                    p_stay = 0.0; p_to_out = 1.0;   % NTN not allowed -> drains out
                else
                    p_stay = pNTN; p_to_out = 1-pNTN;
                end
            end

            i_up=min(N,i+1); i_same=i; i_dn=max(0,i-1);
            p_up=p_arr*(1-p_dep); p_same=p_arr*p_dep + p_no*(1-p_dep); p_dn=p_no*p_dep;

            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i_up,j),  p_stay*p_up);
            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i_same,j),p_stay*p_same);
            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i_dn,j),  p_stay*p_dn);

            if p_to_out>0
                [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i_up,3),  p_to_out*p_up);
                [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i_same,3),p_to_out*p_same);
                [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i_dn,3),  p_to_out*p_dn);
            end
        else
            if force_tn,      p_to_TN = 1; p_to_NTN = 0;
            elseif force_ntn, p_to_TN = 0; p_to_NTN = 1;
            else,             p_to_TN = 1-pNTN; p_to_NTN = pNTN;
            end

            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(min(N,i+1),3),(1-pRec)*p_arr);
            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i,3),         (1-pRec)*p_no);

            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(min(N,i+1),1),pRec*p_to_TN*p_arr);
            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i,1),         pRec*p_to_TN*p_no);

            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(min(N,i+1),2),pRec*p_to_NTN*p_arr);
            [I,J,V,nz]=add_entry(I,J,V,nz,s,state_id(i,2),         pRec*p_to_NTN*p_no);
        end
    end
end

I=I(1:nz); J=J(1:nz); V=V(1:nz);
end

function p_dep = departure_probability(i, mu_i, mode)
if i==0, p_dep=0; return; end
if strcmpi(mode,'per_device')
    p_dep = min(1, mu_i/i);
elseif strcmpi(mode,'aggregate')
    p_dep = 1 - exp(-max(0,mu_i)/i);
    p_dep = min(1,max(0,p_dep));
else
    error('Unsupported mk_service_mode');
end
end

function Pn = normalize_sparse_rows(P)
rs = full(sum(P,2));
[nr,~]=size(P);
for r=1:nr
    if rs(r)<=0
        P(r,r)=1;
    else
        P(r,:)=P(r,:)/rs(r);
    end
end
Pn=P;
end

function [pi_stat, converged, iters] = stationary_direct_sparse(P)
S = size(P,1);
A = (P' - speye(S));
A(end,:) = 1;                 % normalization equation
b = sparse(S,1);
b(end) = 1;

lastwarn('');              % reset warning buffer
x = A \ b;                 % solve
[warnMsg, warnId] = lastwarn;

if ~isempty(warnId)
    converged = false;
    iters = 0;
    pi_stat = nan(1,S);
    return;
end

pi_stat = full(x)';           % row vector
pi_stat = max(0, pi_stat);
s = sum(pi_stat);
if s <= 0 || any(~isfinite(pi_stat))
    converged = false;
    iters = 0;
    pi_stat = nan(1,S);
    return;
end
pi_stat = pi_stat / s;
converged = true;
iters = 1;                    % direct solve
end

function [pi_stat, converged, iters] = stationary_power_iteration(P, tol, maxit)
S=size(P,1); pi_stat=ones(1,S)/S; converged=false; iters=maxit;
for k=1:maxit
    pin = pi_stat*P;
    if norm(pin-pi_stat,1)<tol
        pi_stat=pin; converged=true; iters=k; break;
    end
    pi_stat=pin;
end
pi_stat = max(0,pi_stat);
s = sum(pi_stat);
if s <= 0 || any(~isfinite(pi_stat))
    converged = false;
    iters = maxit;
    pi_stat = nan(1,S);
    return;
end
pi_stat = pi_stat / s;
end

function [D,L] = compute_eq10_eq11_from_pi(pi_stat,mk,Ps_TN,Ps_NTN,mu_TN,mu_NTN)
N=mk.N; piOUT=0; Dsum=0; succSum=0;
for i=0:N
    idTN=state_id(i,1); idNTN=state_id(i,2); idOUT=state_id(i,3);
    if mu_TN(i+1)>0,  Dsum = Dsum + pi_stat(idTN)*(i/mu_TN(i+1)); end
    if mu_NTN(i+1)>0, Dsum = Dsum + pi_stat(idNTN)*(i/mu_NTN(i+1)); end
    piOUT = piOUT + pi_stat(idOUT);
    succSum = succSum + pi_stat(idTN)*Ps_TN(i+1) + pi_stat(idNTN)*Ps_NTN(i+1);
end
D = min(mk.D_max,max(0,Dsum + piOUT*mk.T_OFF));
L = max(0,min(1,1-succSum));
end

function Pd = compute_Pd_sg_exact(gamma0,sigma2,Pj,Omegaj,rho,alpha)
s = gamma0/(Pj*Omegaj);
LI = exp(-pi*rho*gamma(1-2/alpha)*(s*Pj)^(2/alpha));
Pd = exp(-(gamma0*sigma2)/(Pj*Omegaj))*LI;
Pd = max(0,min(1,Pd));
end

function [I,J,V,nz] = add_entry(I,J,V,nz,r,c,val)
if val==0, return; end
nz=nz+1; I(nz)=r; J(nz)=c; V(nz)=val;
end

function id = state_id(i,j)
id = i*3 + j;
end

function make_K_sensitivity_figure(results_sensitivity, ordK, Kvals, Ns_list, EDs, metric_name)
% metric_name in {'energy','delay','loss'}

figure('Name',sprintf('K_NTN Sensitivity - %s',upper(metric_name)), ...
    'NumberTitle','on','Color','w','Position',[120 120 1600 650]);
tiledlayout(1, numel(Ns_list), 'Padding','compact', 'TileSpacing','compact');

for iM = 1:numel(Ns_list)
    nexttile; hold on; grid on; box on;
    set(gca,'FontSize',21);
    M = Ns_list(iM);

    for iE = 1:numel(EDs)
        y_atlb = zeros(size(Kvals));
        y_ahsa = zeros(size(Kvals));

        for kk = 1:numel(Kvals)
            rs = results_sensitivity(ordK(kk));
            switch lower(metric_name)
                case 'energy'
                    y_atlb(kk) = rs.ALB_energy_mean(iM,iE);
                    y_ahsa(kk) = rs.AHSA_energy_mean(iM,iE);
                case 'delay'
                    y_atlb(kk) = rs.ALB_delay_mean(iM,iE);
                    y_ahsa(kk) = rs.AHSA_delay_mean(iM,iE);
                case 'loss'
                    y_atlb(kk) = rs.ALB_loss_mean(iM,iE);
                    y_ahsa(kk) = rs.AHSA_loss_mean(iM,iE);
            end
        end

        % solid = ATLB, dashed = AHSA for same ED
        plot(Kvals, y_atlb, '-o', 'LineWidth',1.8, 'MarkerSize',5, ...
            'DisplayName', sprintf('ATLB N=%d', EDs(iE)));
        plot(Kvals, y_ahsa, '--s', 'LineWidth',1.4, 'MarkerSize',4, ...
            'HandleVisibility','off');
    end

    xlabel('K_{NTN} (dB)', 'FontSize',21);
    switch lower(metric_name)
        case 'energy', ylabel('Energy [mJ]', 'FontSize',21);
        case 'delay',  ylabel('Delay [slots]', 'FontSize',21);
        case 'loss',   ylabel('Loss probability', 'FontSize',21);
    end
    title(sprintf('M = %d', M), 'FontSize',21);
end

lg = legend('Location','eastoutside', 'FontSize',21);
title(lg, 'Solid: ATLB, Dashed: AHSA');
lg.Title.FontSize = 21;


end

function make_rayleigh_match_figure_combined(EDs, Ns_list, ...
    sim_delay_TN, markov_delay_TN, sim_delay_NTN, markov_delay_NTN, ...
    sim_loss_TN, markov_loss_TN, sim_loss_NTN, markov_loss_NTN)
% Combines TN/NTN delay and loss into a single 2x2 figure.

panels = {
    sim_delay_TN,  markov_delay_TN,  'Delay [slots]',    'Delay — TN-only';
    sim_delay_NTN, markov_delay_NTN, 'Delay [slots]',    'Delay — NTN-only';
    sim_loss_TN,   markov_loss_TN,   'Loss probability', 'Loss — TN-only';
    sim_loss_NTN,  markov_loss_NTN,  'Loss probability', 'Loss — NTN-only'};

figure('Name','Rayleigh Matched Validation (Combined)', ...
    'NumberTitle','on','Color','w','Position',[100 100 1300 950]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

for p = 1:4
    nexttile; hold on; grid on; box on;
    set(gca,'FontSize',21);
    sim_mat = panels{p,1}; markov_mat = panels{p,2};
    for iM = 1:numel(Ns_list)
        plot(EDs, sim_mat(iM,:), '-o', 'LineWidth',1.8, 'MarkerSize',5, ...
            'DisplayName', sprintf('MC, M=%d', Ns_list(iM)));
        plot(EDs, markov_mat(iM,:), '--s', 'LineWidth',1.6, 'MarkerSize',5, ...
            'DisplayName', sprintf('Markov, M=%d', Ns_list(iM)));
    end
    xlabel('Number of EDs','FontSize',21); ylabel(panels{p,3},'FontSize',21);
    title(panels{p,4}, 'FontSize',21);
    xticks([0 1000 2000 3000 4000 5000]);
    legend('Location','best','FontSize',21);
end


fname = 'fig_rayleigh_match_combined.png';
exportgraphics(gcf, fname, 'Resolution', 300);
fprintf('[Saved] %s\n', fname);

end


function make_param_sensitivity_tripanel(results_delta, results_Bmin, results_tauSINR, ...
    delta_vals, Bmin_vals, tau_vals, EDs, Nsats_list, M_idx, metric_name)

figure('Name',sprintf('ATLB Sensitivity (%s) - M=%d', upper(metric_name), Nsats_list(M_idx)), ...
    'NumberTitle','on','Color','w','Position',[100 100 1600 550]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

% Panel 1: delta
nexttile; hold on; grid on; box on;
set(gca,'FontSize',21);
for v = 1:numel(delta_vals)
    [y, ci] = pick_metric(results_delta(v), metric_name, M_idx);
    errorbar(EDs, y, ci, '-o', 'LineWidth',1.2, 'DisplayName', sprintf('\\delta=%.2f', delta_vals(v)));
end
xlabel('Number of EDs','FontSize',21); ylabel(ylabel_from_metric(metric_name),'FontSize',21);
title(sprintf('\\delta sweep (M=%d)', Nsats_list(M_idx)), 'FontSize',21); legend('Location','best','FontSize',21);

% Panel 2: Bmin
nexttile; hold on; grid on; box on;
set(gca,'FontSize',21);
for v = 1:numel(Bmin_vals)
    [y, ci] = pick_metric(results_Bmin(v), metric_name, M_idx);
    errorbar(EDs, y, ci, '-o', 'LineWidth',1.2, 'DisplayName', sprintf('B_{min}=%.2f', Bmin_vals(v)));
end
xlabel('Number of EDs','FontSize',21); ylabel(ylabel_from_metric(metric_name),'FontSize',21);
title(sprintf('B_{min} sweep (M=%d)', Nsats_list(M_idx)), 'FontSize',21); legend('Location','best','FontSize',21);

% Panel 3: tauSINR
nexttile; hold on; grid on; box on;
set(gca,'FontSize',21);
for v = 1:numel(tau_vals)
    [y, ci] = pick_metric(results_tauSINR(v), metric_name, M_idx);
    errorbar(EDs, y, ci, '-o', 'LineWidth',1.2, 'DisplayName', sprintf('\\tau_{SINR}=%g dB', tau_vals(v)));
end
xlabel('Number of EDs','FontSize',21); ylabel(ylabel_from_metric(metric_name),'FontSize',21);
title(sprintf('\\tau_{SINR} sweep (M=%d)', Nsats_list(M_idx)), 'FontSize',21); legend('Location','best','FontSize',21);
end

function [y, ci] = pick_metric(s, metric_name, M_idx)
switch lower(metric_name)
    case 'delay'
        y = s.ALB_delay_mean(M_idx,:);
        if isfield(s,'ALB_delay_ci')
            ci = s.ALB_delay_ci(M_idx,:);
        else
            ci = zeros(size(y));
        end
    case 'offload'
        y = s.ALB_off_mean(M_idx,:);
        if isfield(s,'ALB_off_ci')
            ci = s.ALB_off_ci(M_idx,:);
        else
            ci = zeros(size(y));
        end
    otherwise
        error('Unsupported metric_name: %s', metric_name);
end
end

function ylab = ylabel_from_metric(metric_name)
switch lower(metric_name)
    case 'delay'
        ylab = 'ATLB delay [slots]';
    case 'offload'
        ylab = 'NTN usage fraction';
    otherwise
        ylab = 'Metric';
end
elapsed = toc;  % seconds since tic
fprintf('Simulation finished in %.3f seconds (%.2f minutes)\n', elapsed, elapsed/60);
end