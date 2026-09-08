'use client';

import React, { useState, useEffect } from 'react';
import {
  Swords,
  Play,
  Zap,
  Clock,
  DollarSign,
  Award,
  AlertCircle,
  CheckCircle2,
  Trophy,
  Scale,
  ThumbsUp,
  RefreshCw,
  Sliders,
  Flame,
} from 'lucide-react';
import {
  ModelMetadata,
  fetchModels,
  compareInArena,
  ArenaComparisonResponse,
  ModelEloRecord,
  JudgeVerdict,
  fetchEloLeaderboard,
  runArenaBattle,
  voteInArena,
  runArenaTournament,
} from '@/lib/api';

export default function ArenaView() {
  const [activeTab, setActiveTab] = useState<'arena' | 'leaderboard'>('arena');
  const [availableModels, setAvailableModels] = useState<ModelMetadata[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'libra-llama-tied',
    'libra-mock-v1',
  ]);
  const [prompt, setPrompt] = useState('Explain how attention works in transformers in 2 clear sentences.');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(128);
  const [loading, setLoading] = useState(false);
  const [arenaResult, setArenaResult] = useState<ArenaComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Judging & Voting state
  const [battleModelA, setBattleModelA] = useState<string>('');
  const [battleModelB, setBattleModelB] = useState<string>('');
  const [judgeVerdict, setJudgeVerdict] = useState<JudgeVerdict | null>(null);
  const [judgeLoading, setJudgeLoading] = useState(false);
  const [voteMessage, setVoteMessage] = useState<string | null>(null);

  // Leaderboard state
  const [leaderboard, setLeaderboard] = useState<ModelEloRecord[]>([]);
  const [leaderboardCategory, setLeaderboardCategory] = useState<string>('overall');
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);
  const [tournamentRunning, setTournamentRunning] = useState(false);
  const [tournamentSummary, setTournamentSummary] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const data = await fetchModels();
      setAvailableModels(data);
      if (data.length >= 2) {
        setSelectedModels([data[0].id, data[1].id]);
        setBattleModelA(data[0].id);
        setBattleModelB(data[1].id);
      }
    }
    load();
    loadLeaderboard('overall');
  }, []);

  const loadLeaderboard = async (cat: string) => {
    setLeaderboardLoading(true);
    try {
      const data = await fetchEloLeaderboard(cat);
      setLeaderboard(data);
    } catch {
      // ignore
    } finally {
      setLeaderboardLoading(false);
    }
  };

  const handleToggleModel = (id: string) => {
    if (selectedModels.includes(id)) {
      if (selectedModels.length <= 1) return;
      setSelectedModels(selectedModels.filter((m) => m !== id));
    } else {
      if (selectedModels.length >= 4) return;
      setSelectedModels([...selectedModels, id]);
    }
  };

  const handleRunComparison = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || loading || selectedModels.length === 0) return;

    setLoading(true);
    setError(null);
    setJudgeVerdict(null);
    setVoteMessage(null);

    try {
      const targets = selectedModels.map((mId) => {
        const found = availableModels.find((m) => m.id === mId);
        return {
          model: mId,
          provider: found ? found.provider : undefined,
        };
      });

      const res = await compareInArena(prompt, targets, temperature, maxTokens);
      setArenaResult(res);

      if (res.results.length >= 2) {
        setBattleModelA(res.results[0].model);
        setBattleModelB(res.results[1].model);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleAutoJudge = async () => {
    if (!battleModelA || !battleModelB || battleModelA === battleModelB) {
      setError('Please select two distinct models for battle evaluation.');
      return;
    }
    setJudgeLoading(true);
    setError(null);
    setVoteMessage(null);

    try {
      const mA = availableModels.find((m) => m.id === battleModelA);
      const mB = availableModels.find((m) => m.id === battleModelB);

      const res = await runArenaBattle(
        prompt,
        { model: battleModelA, provider: mA?.provider },
        { model: battleModelB, provider: mB?.provider }
      );
      setJudgeVerdict(res.verdict);
      setVoteMessage(
        `AI Referee awarded match to ${
          res.verdict.winner === 'tie'
            ? 'Tie'
            : res.verdict.winner === 'model_a'
            ? battleModelA
            : battleModelB
        }! Rating change: ${battleModelA} (${res.match.delta_a > 0 ? '+' : ''}${res.match.delta_a}), ${battleModelB} (${res.match.delta_b > 0 ? '+' : ''}${res.match.delta_b})`
      );
      await loadLeaderboard(leaderboardCategory);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setJudgeLoading(false);
    }
  };

  const handleManualVote = async (winner: 'model_a' | 'model_b' | 'tie') => {
    if (!battleModelA || !battleModelB || battleModelA === battleModelB) return;
    try {
      const res = await voteInArena(battleModelA, battleModelB, winner, 'overall', prompt);
      const winnerName =
        winner === 'tie' ? 'Tie' : winner === 'model_a' ? battleModelA : battleModelB;
      setVoteMessage(
        `Recorded human vote for ${winnerName}! New ratings: ${battleModelA} (${res.model_a_rating}), ${battleModelB} (${res.model_b_rating})`
      );
      await loadLeaderboard(leaderboardCategory);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRunTournament = async () => {
    if (availableModels.length < 2) return;
    setTournamentRunning(true);
    setTournamentSummary(null);
    try {
      const targets = availableModels.slice(0, 3).map((m) => ({
        model: m.id,
        provider: m.provider,
      }));
      const res = await runArenaTournament(targets, ['coding', 'reasoning', 'factual', 'instruction'], 1);
      setTournamentSummary(
        `Tournament complete! Ran ${res.total_matches} pairwise matches across domains (${res.categories_evaluated.join(', ')}). Elo leaderboard updated.`
      );
      await loadLeaderboard(leaderboardCategory);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setTournamentRunning(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-y-auto p-4 sm:p-8">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-5 gap-4">
          <div>
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
              <Swords className="w-5 h-5 text-indigo-400" />
              <span>Model Comparison Arena & Elo Leaderboard</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Continuous benchmarking, position-bias mitigated automated referee, and Bradley-Terry Elo ratings.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
              <button
                type="button"
                onClick={() => setActiveTab('arena')}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  activeTab === 'arena'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Swords className="w-3.5 h-3.5" />
                <span>Side-by-Side & Battle</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setActiveTab('leaderboard');
                  loadLeaderboard(leaderboardCategory);
                }}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  activeTab === 'leaderboard'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Trophy className="w-3.5 h-3.5" />
                <span>Elo Leaderboard</span>
              </button>
            </div>
          </div>
        </div>

        {/* TAB 1: SIDE-BY-SIDE & BATTLE */}
        {activeTab === 'arena' && (
          <>
            {/* Configuration Bar */}
            <form
              onSubmit={handleRunComparison}
              className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm"
            >
              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  Select Contending Models (Choose 2 to 4)
                </label>
                <div className="flex flex-wrap gap-2">
                  {availableModels.map((m) => {
                    const isChecked = selectedModels.includes(m.id);
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => handleToggleModel(m.id)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs border transition-all ${
                          isChecked
                            ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200 font-medium'
                            : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isChecked ? 'bg-indigo-400' : 'bg-slate-600'
                          }`}
                        />
                        <span>{m.name}</span>
                        <span className="text-[10px] text-slate-400 font-mono">({m.provider})</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  Benchmark Prompt
                </label>
                <textarea
                  rows={2}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter a prompt to evaluate across all selected models..."
                  className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl p-3 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all resize-none shadow-inner"
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-800/80">
                <div className="flex items-center gap-6 text-xs text-slate-300">
                  <div className="flex items-center gap-2">
                    <span>Temp:</span>
                    <input
                      type="number"
                      step="0.1"
                      min="0.0"
                      max="1.5"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-16 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-right text-indigo-300"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <span>Max Tokens:</span>
                    <input
                      type="number"
                      step="32"
                      min="32"
                      max="512"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                      className="w-20 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-right text-indigo-300"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading || selectedModels.length === 0}
                  className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium px-5 py-2 rounded-xl text-xs transition-all shadow-md shadow-indigo-600/20"
                >
                  {loading ? (
                    <>
                      <span className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                      <span>Benchmarking...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>Execute Arena Run</span>
                    </>
                  )}
                </button>
              </div>
            </form>

            {error && (
              <div className="flex items-center gap-2 bg-rose-950/40 border border-rose-800/60 text-rose-300 px-4 py-3 rounded-xl text-xs">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {voteMessage && (
              <div className="flex items-center gap-2 bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 px-4 py-3 rounded-xl text-xs">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>{voteMessage}</span>
              </div>
            )}

            {/* Results Grid */}
            {arenaResult && (
              <div className="space-y-6 animate-in fade-in duration-200">
                {/* Leaderboard Awards */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
                      <Clock className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-400">Fastest TTFT</div>
                      <div className="text-sm font-bold text-slate-100 truncate">
                        {arenaResult.rankings.fastest_ttft || 'N/A'}
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0">
                      <Zap className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-400">Highest Throughput</div>
                      <div className="text-sm font-bold text-slate-100 truncate">
                        {arenaResult.rankings.highest_throughput || 'N/A'}
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0">
                      <DollarSign className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-400">Most Economical</div>
                      <div className="text-sm font-bold text-slate-100 truncate">
                        {arenaResult.rankings.lowest_cost || 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Outputs Side-by-Side */}
                <div
                  className={`grid grid-cols-1 md:grid-cols-${Math.min(
                    arenaResult.results.length,
                    3
                  )} gap-4`}
                >
                  {arenaResult.results.map((r) => (
                    <div
                      key={r.model}
                      className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden flex flex-col justify-between"
                    >
                      <div>
                        {/* Model Header */}
                        <div className="p-4 border-b border-slate-800/80 bg-slate-950/40 flex items-center justify-between">
                          <div>
                            <h4 className="font-semibold text-slate-100 text-sm truncate">
                              {r.model}
                            </h4>
                            <p className="text-[10px] text-slate-400 uppercase tracking-wider">
                              {r.provider}
                            </p>
                          </div>
                          {r.success ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          ) : (
                            <AlertCircle className="w-4 h-4 text-rose-400" />
                          )}
                        </div>

                        {/* Output Text */}
                        <div className="p-4 text-xs text-slate-200 leading-relaxed min-h-[120px] whitespace-pre-wrap">
                          {r.success ? (
                            r.output_text || (
                              <span className="text-slate-500 italic">No output generated</span>
                            )
                          ) : (
                            <span className="text-rose-400">{r.error}</span>
                          )}
                        </div>
                      </div>

                      {/* Telemetry Footer */}
                      <div className="p-3 bg-slate-950/70 border-t border-slate-800 text-[11px] grid grid-cols-2 gap-2 text-slate-400">
                        <div>
                          <span>TTFT: </span>
                          <span className="text-slate-200 font-mono">
                            {r.ttft_ms !== null ? `${r.ttft_ms.toFixed(1)}ms` : 'N/A'}
                          </span>
                        </div>
                        <div>
                          <span>Latency: </span>
                          <span className="text-slate-200 font-mono">
                            {r.total_latency_ms.toFixed(1)}ms
                          </span>
                        </div>
                        <div>
                          <span>Speed: </span>
                          <span className="text-indigo-300 font-mono font-medium">
                            {r.tokens_per_second.toFixed(1)} tok/s
                          </span>
                        </div>
                        <div>
                          <span>Cost: </span>
                          <span className="text-emerald-400 font-mono font-medium">
                            {r.is_free ? '$0.00' : `$${r.cost_usd.toFixed(6)}`}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Head-to-Head Arena Battle & Judging Section */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                    <div>
                      <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                        <Scale className="w-4 h-4 text-indigo-400" />
                        <span>Head-to-Head Referee & Elo Updates</span>
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Compare two candidate models, trigger dual-pass position-bias mitigated auto-judging, or cast a human vote.
                      </p>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <select
                        value={battleModelA}
                        onChange={(e) => setBattleModelA(e.target.value)}
                        className="bg-slate-950 border border-slate-800 text-slate-200 rounded-lg px-2.5 py-1 text-xs"
                      >
                        {selectedModels.map((m) => (
                          <option key={`a-${m}`} value={m}>
                            Model A: {m}
                          </option>
                        ))}
                      </select>
                      <span className="text-slate-500 font-bold">vs</span>
                      <select
                        value={battleModelB}
                        onChange={(e) => setBattleModelB(e.target.value)}
                        className="bg-slate-950 border border-slate-800 text-slate-200 rounded-lg px-2.5 py-1 text-xs"
                      >
                        {selectedModels.map((m) => (
                          <option key={`b-${m}`} value={m}>
                            Model B: {m}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      disabled={judgeLoading || battleModelA === battleModelB}
                      onClick={handleAutoJudge}
                      className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium px-4 py-2 rounded-xl text-xs transition-all shadow-sm"
                    >
                      {judgeLoading ? (
                        <>
                          <span className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                          <span>Referee Evaluating...</span>
                        </>
                      ) : (
                        <>
                          <Award className="w-3.5 h-3.5" />
                          <span>AI Referee (Auto-Judge)</span>
                        </>
                      )}
                    </button>

                    <div className="h-6 w-px bg-slate-800 hidden sm:block" />

                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 font-medium">Human Vote:</span>
                      <button
                        type="button"
                        onClick={() => handleManualVote('model_a')}
                        className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg text-xs transition-all"
                      >
                        <ThumbsUp className="w-3 h-3 text-indigo-400" />
                        <span>Vote {battleModelA}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleManualVote('model_b')}
                        className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg text-xs transition-all"
                      >
                        <ThumbsUp className="w-3 h-3 text-emerald-400" />
                        <span>Vote {battleModelB}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleManualVote('tie')}
                        className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg text-xs transition-all"
                      >
                        Tie
                      </button>
                    </div>
                  </div>

                  {/* Verdict Details Panel */}
                  {judgeVerdict && (
                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3 mt-3">
                      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                        <div className="flex items-center gap-2">
                          <Trophy className="w-4 h-4 text-amber-400" />
                          <span className="text-xs font-bold text-slate-100">
                            Verdict:{' '}
                            <span className="text-indigo-400">
                              {judgeVerdict.winner === 'tie'
                                ? 'Tie'
                                : judgeVerdict.winner === 'model_a'
                                ? battleModelA
                                : battleModelB}
                            </span>
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] bg-emerald-950/60 border border-emerald-800/50 text-emerald-300 px-2 py-0.5 rounded-md font-mono">
                            Dual-Pass Swapped
                          </span>
                          <span className="text-xs text-slate-400 font-mono">
                            Conf: {(judgeVerdict.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                          <div className="font-semibold text-slate-300 truncate mb-1">
                            {battleModelA}
                          </div>
                          <div className="text-lg font-bold text-indigo-300">
                            {judgeVerdict.score_a.toFixed(1)}{' '}
                            <span className="text-xs text-slate-500 font-normal">/ 10.0</span>
                          </div>
                        </div>

                        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                          <div className="font-semibold text-slate-300 truncate mb-1">
                            {battleModelB}
                          </div>
                          <div className="text-lg font-bold text-emerald-300">
                            {judgeVerdict.score_b.toFixed(1)}{' '}
                            <span className="text-xs text-slate-500 font-normal">/ 10.0</span>
                          </div>
                        </div>
                      </div>

                      <div className="text-xs text-slate-300 bg-slate-900/40 p-3 rounded-lg border border-slate-800/60 leading-relaxed">
                        <span className="font-semibold text-slate-400">Rationale: </span>
                        {judgeVerdict.rationale}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {/* TAB 2: ELO LEADERBOARD */}
        {activeTab === 'leaderboard' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/80 border border-slate-800 p-4 rounded-2xl">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Domain Filter:
                </span>
                <div className="flex flex-wrap gap-1">
                  {['overall', 'coding', 'reasoning', 'factual', 'instruction'].map((cat) => (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => {
                        setLeaderboardCategory(cat);
                        loadLeaderboard(cat);
                      }}
                      className={`px-3 py-1 rounded-lg text-xs capitalize transition-all ${
                        leaderboardCategory === cat
                          ? 'bg-indigo-600 text-white font-medium'
                          : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={tournamentRunning || leaderboardLoading}
                  onClick={handleRunTournament}
                  className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium px-4 py-1.5 rounded-xl text-xs transition-all shadow-sm"
                >
                  {tournamentRunning ? (
                    <>
                      <span className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                      <span>Running Tournament...</span>
                    </>
                  ) : (
                    <>
                      <Flame className="w-3.5 h-3.5" />
                      <span>Run Round-Robin Tournament</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => loadLeaderboard(leaderboardCategory)}
                  className="p-1.5 bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 rounded-xl transition-all"
                  title="Refresh Leaderboard"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {tournamentSummary && (
              <div className="flex items-center gap-2 bg-indigo-950/40 border border-indigo-800/60 text-indigo-300 px-4 py-3 rounded-xl text-xs">
                <CheckCircle2 className="w-4 h-4 text-indigo-400 shrink-0" />
                <span>{tournamentSummary}</span>
              </div>
            )}

            {/* Leaderboard Table */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950/80 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-4">Rank</th>
                      <th className="py-3 px-4">Model ID</th>
                      <th className="py-3 px-4">Elo Rating</th>
                      <th className="py-3 px-4">95% CI</th>
                      <th className="py-3 px-4">Win Rate</th>
                      <th className="py-3 px-4">Matches (W-L-T)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {leaderboard.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="py-8 text-center text-slate-500">
                          {leaderboardLoading
                            ? 'Loading ratings...'
                            : 'No matches recorded yet. Run a battle or automated tournament above!'}
                        </td>
                      </tr>
                    ) : (
                      leaderboard.map((item, idx) => {
                        const margin = Math.round((item.ci_upper - item.ci_lower) / 2);
                        return (
                          <tr key={item.model_id} className="hover:bg-slate-800/30 transition-colors">
                            <td className="py-3 px-4 font-mono font-bold text-slate-400">
                              {idx === 0 ? (
                                <span className="text-amber-400 flex items-center gap-1">
                                  <Trophy className="w-3.5 h-3.5" /> 1
                                </span>
                              ) : idx === 1 ? (
                                <span className="text-slate-300">2</span>
                              ) : idx === 2 ? (
                                <span className="text-amber-600">3</span>
                              ) : (
                                idx + 1
                              )}
                            </td>
                            <td className="py-3 px-4 font-semibold text-slate-100">
                              {item.model_id}
                            </td>
                            <td className="py-3 px-4 font-mono font-bold text-indigo-300">
                              {item.rating.toFixed(1)}
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-400">
                              ±{margin} <span className="text-[10px] text-slate-500">[{item.ci_lower}, {item.ci_upper}]</span>
                            </td>
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className="bg-emerald-500 h-full rounded-full"
                                    style={{ width: `${item.win_rate}%` }}
                                  />
                                </div>
                                <span className="font-mono text-xs text-slate-200">
                                  {item.win_rate.toFixed(1)}%
                                </span>
                              </div>
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-400">
                              {item.matches} ({item.wins}-{item.losses}-{item.ties})
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
