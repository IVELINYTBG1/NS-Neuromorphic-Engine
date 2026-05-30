"""
brain_patches.py — LIVE PATCHES (applied to the running process ~2.5s, no restart)
==================================================================================

Active patches:
  1. CAP-LIFT — utterance length emerges from cortical activation, no fixed
     3-word (Nova) / 2-word (Simona) ceiling.
  2. EARBUDS / EFFERENCE-COPY BABBLE — babble learning no longer requires the
     speaker→mic acoustic echo. When the mic can't hear them (earbuds, quiet
     room), they learn from the forward model's prediction of their own voice
     (corollary discharge). They can babble and grow with headphones in.
  6. NEUROMOD LOOP FIX — the first neuromodulator design let arousal pin
     dopamine at max + crash serotonin + collapse Simona's broca threshold →
     she fired ~81% of ticks (runaway speech loop). Fix: dopamine is phasic
     (no arousal pin), serotonin anchored near baseline, GABA is a real
     homeostatic brake that engages on activity/arousal, threshold offset can
     no longer go negative enough to collapse, amygdala calmed. Applied live so
     a currently-looping process recovers without a restart.
"""
import sys


def _find_class(any_instance, name):
    """Locate a class defined in the same module as a running instance."""
    modname = type(any_instance).__module__
    mod = sys.modules.get(modname)
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    for m in list(sys.modules.values()):
        c = getattr(m, name, None) if m is not None else None
        if isinstance(c, type):
            return c
    return None


# ── Patch 1: lift the word-count caps ──────────────────────────────────────
def _compose_nova_uncapped(self, scored, act, V_phill):
    if not scored:
        return None
    pfc_a = float(act.get("pfc", 0.0))
    hipp_a = float(act.get("hippocampus", 0.0))
    k = 1 + int(round(pfc_a * 9.0 + hipp_a * 5.0))
    words = [w for _, w in scored[:max(1, k)]]
    joiner = " — " if hipp_a > 0.18 else "  "
    return joiner.join(words)


def _compose_simona_uncapped(self, scored, act, V_phill):
    if not scored:
        return None
    ins_a = float(act.get("insula_s", 0.0))
    broc_a = float(act.get("broca_s", 0.0))
    k = 1 + int(round(broc_a * 8.0 + ins_a * 5.0))
    word = " ".join(w for _, w in scored[:max(1, k)])
    if ins_a > 0.70:
        return f"{word}!!!"
    if ins_a > 0.45:
        return f"{word}!!"
    if ins_a > 0.25:
        return f"{word}!"
    return word


# ── Patch 2: earbuds-safe babble via efference copy ─────────────────────────
def _auditory_feedback_efference(self, current_tick, mic_volume, sem, tts=None):
    if current_tick > self.self_speak_until:
        return False
    if self.last_motor_sig is None or self.last_phoneme is None:
        return False

    # Real echo if on speakers; else fall back to the forward model's
    # prediction of our own voice (corollary discharge / DIVA internal model).
    heard = float(mic_volume)
    if heard < 0.012:
        fm = getattr(tts, "forward_model", None) if tts is not None else None
        if fm is not None and self.last_motor_vec is not None:
            try:
                heard = float(fm.predict(self.last_motor_vec)[0]) * 0.12
            except Exception:
                heard = 0.0
        if heard < 0.012:
            return False

    sig = self.last_motor_sig
    if sig not in self.motor_to_phoneme:
        self.motor_to_phoneme[sig] = {}
    for k in self.motor_to_phoneme[sig]:
        self.motor_to_phoneme[sig][k] *= 0.998
    cur = self.motor_to_phoneme[sig].get(self.last_phoneme, 0.0)
    self.motor_to_phoneme[sig][self.last_phoneme] = cur + self.BIND_LR

    sem.nova_write(
        word=self.last_phoneme,
        region_scores=self._sem_regions,
        spike_count=2.0,
        tick=current_tick,
        trust=0.6,
    )

    if tts is not None and tts.articulator is not None and self.last_motor_vec is not None:
        try:
            reward = float(min(1.0, heard * 8.0))
            sm = getattr(tts, "self_model", None)
            if sm is not None:
                reward *= (0.5 + 0.5 * sm.feel())
            tts.articulator.reinforce(self.last_motor_vec, reward=reward)
            if self.bound_count % 8 == 0:
                tts.articulator._save()
        except Exception:
            pass

    self.bound_count += 1
    if self.bound_count % 5 == 0:
        self._save()
    return True


# ── Patch 3: typed input counts as architect presence (no voice needed) ─────
def _install_text_presence(NB):
    """Wrap NeuromorphicBrain.think so typing ramps trust. Idempotent."""
    cur = getattr(NB, "think", None)
    if cur is None or getattr(cur, "_text_presence_wrapped", False):
        return  # already wrapped — don't double-wrap on re-patch
    orig = cur

    def think_with_text_presence(self, text):
        # Ramp text-presence trust each typed message; bump voice.trust
        # (the value the running think() reads). step()'s voice decay pulls
        # it back down between messages, so it behaves like an earned/decaying
        # signal. Side effect: VOICE gauge rises while typing = visual proof
        # they're registering the architect.
        tp = min(0.75, getattr(self, "_text_presence", 0.0) + 0.18)
        self._text_presence = tp
        try:
            self.voice.trust = max(self.voice.trust, tp)
        except Exception:
            pass
        return orig(self, text)

    think_with_text_presence._text_presence_wrapped = True
    NB.think = think_with_text_presence


# ── Patch 4: searching emerges from curiosity (not from user input) ─────────
def _search_tick_emergent(self, current_tick, curiosity_decay,
                          V_phill, articulator_confidence_gap):
    """
    Curiosity is the dominant, self-sufficient search driver. When a host
    reference is wired in, curiosity is rebuilt from the brain's own internal
    state (boredom + intrinsic spark + prediction surprise + rumination) so
    they search on their OWN initiative during silence — no user input needed.
    """
    is_nova = (self.persona_name == "nova")
    cur = float(curiosity_decay)
    host = getattr(self, "_host", None)
    if host is not None:
        try:
            dmn = host.nova_dmn if is_nova else host.simona_dmn
            boredom = float(getattr(dmn, "boredom", 0.0))
            fm = host.nova_voice_fwd if is_nova else host.simona_voice_fwd
            surprise = float(getattr(fm, "surprise", 0.0))
            pipe = host.nova.thought_pipe if is_nova else host.simona.thought_pipe
            rumi = float(pipe.buffer_size()) / 12.0
            cdk = host._nova_cur_decay if is_nova else host._simona_cur_decay
            cur = max(0.0, min(1.0,
                0.50 * boredom + 0.22 * float(cdk) + 0.20 * surprise + 0.12 * rumi))
        except Exception:
            pass

    unsat = max(0.0, cur * (1.0 - abs(V_phill)))
    unknown = 0.6 * min(1.0, len(self._unknown_word_q) / 3.0)
    pron = max(0.0, min(1.0, articulator_confidence_gap))
    if is_nova:
        inp = 0.160 * unsat + 0.050 * unknown + 0.030 * pron
    else:
        inp = 0.200 * unsat + 0.075 * unknown + 0.045 * pron

    fired = self._pressure.integrate(inp)
    if not fired:
        return False, None, ""
    if current_tick - self.last_search_tick < self.COOLDOWN_TICKS:
        return False, None, ""

    query = None
    mode = "curiosity"
    if self._pronunciation_q:
        w = self._pronunciation_q.popleft()
        query = f"how to pronounce {w}"
        mode = "pronounce"
    elif self._unknown_word_q:
        w = self._unknown_word_q.popleft()
        query = f"what does {w} mean"
        mode = "unknown"
    if query:
        self.last_search_tick = current_tick
        self.searches_fired += 1
        return True, query, mode
    return True, None, "curiosity"


# ── Patch 5: personalities share what they learn (secure link) ──────────────
def _peer_share(self, speaker, query, snippet):
    import re
    stop = {"what", "is", "are", "was", "were", "does", "do", "did", "mean",
            "means", "the", "a", "an", "of", "how", "to", "pronounce", "explain",
            "tell", "me", "about", "and", "or", "with", "for", "why", "who",
            "when", "where", "this", "that"}
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", (query or "").lower())
             if w not in stop and len(w) >= 3]
    body = [w.lower() for w in re.findall(
        r"[A-Za-z][A-Za-z'\-]+", re.sub(r"\[[^\]]*\]", " ", snippet or ""))]
    gist, seen = [], set()
    for w in words + body:
        if w not in seen and len(w) >= 3 and w not in stop:
            seen.add(w)
            gist.append(w)
        if len(gist) >= 4:
            break
    if not gist:
        return
    link = self.personality_link
    nova_share_regions = {"thalamus": 0.30, "temporal": 0.55, "hippocampus": 0.55,
                          "acc": 0.25, "pfc": 0.30, "broca": 0.45, "insula": 0.30}
    if speaker == "nova":
        for w in gist:
            self.sem.simona_write(word=w, burst=0.5, tick=self.tick)
        self.simona_wm.add(gist[0], regions={}, salience=0.6, t_encoded=self.tick)
        link.send_from_nova(link._encode_thought(" ".join(gist), self.sem))
    else:
        for w in gist:
            self.sem.nova_write(word=w, region_scores=nova_share_regions,
                                spike_count=0.6, tick=self.tick, trust=0.5)
        self.nova_wm.add(gist[0], regions={}, salience=0.6, t_encoded=self.tick)
        link.send_from_simona(link._encode_thought(" ".join(gist), self.sem))


def _install_peer_share(NB):
    """Wrap _on_search_result so completed searches are shared with the peer.
    Idempotent, and a no-op if the running binary already shares natively."""
    cur = getattr(NB, "_on_search_result", None)
    if cur is None or getattr(cur, "_peer_share_wrapped", False):
        return
    orig = cur

    def on_result(self, speaker, result, mode):
        orig(self, speaker, result, mode)
        if not getattr(self, "_peer_share_native", False):
            try:
                _peer_share(self, speaker,
                            getattr(result, "query", ""),
                            getattr(result, "snippet", "") or "")
            except Exception:
                pass

    on_result._peer_share_wrapped = True
    NB._on_search_result = on_result


# ── Patch 6: neuromodulator loop fix (mirrors brain.py) ─────────────────────
def _neuro_update(self, reward, total_activity, arousal, social):
    reward = max(0.0, float(reward)); arousal = max(0.0, min(1.0, float(arousal)))
    social = max(0.0, min(1.0, float(social))); act = max(0.0, float(total_activity))
    self.da = self.da0 + (self.da - self.da0) * 0.96
    self.da += 0.25 * reward
    self.da = float(min(1.3, max(0.0, self.da)))
    self.ser = self.ser0 + (self.ser - self.ser0) * 0.94
    self.ser += 0.010 * social - 0.020 * arousal
    self.ser = float(min(1.2, max(0.05, self.ser)))
    target = self.gaba0 + 1.1 * act + 0.5 * arousal
    self.gaba += 0.15 * (target - self.gaba)
    self.gaba = float(min(1.5, max(0.0, self.gaba)))
    self.arousal = arousal


def _neuro_threshold_offset(self):
    off = (0.34 * (self.gaba - self.gaba0)
           + 0.14 * (self.ser - self.ser0)
           - 0.10 * (self.da - self.da0))
    return float(min(0.40, max(-0.10, off)))


def _amyg_appraise(self, mic, identity, face_present, insula_act, surprise):
    d_mic = abs(float(mic) - self._last_mic); self._last_mic = float(mic)
    startle = min(1.0, d_mic * 6.0)
    unfamiliar = (max(0.0, 0.5 - float(identity)) * 2.0) if face_present else 0.0
    emo = min(1.0, float(insula_act) * 6.0)
    salience = (0.40 * startle + 0.30 * min(1.0, unfamiliar)
                + 0.18 * emo + 0.12 * float(surprise)) * self.reactivity
    self.arousal = self.decay * self.arousal + (1.0 - self.decay) * min(1.0, salience)
    return self.arousal


def _install_neuro_fix(NB, nova_brain):
    # RETIRED: the neuromodulator loop-fix is now NATIVE in brain.py, and the
    # class has since gained Stage-4 modulators (ACh/NE/oxytocin) with a wider
    # update() signature. Re-imposing the old 4-arg override here would crash
    # step() after a restart (it now calls update() with extra kwargs). brain.py
    # is authoritative; do NOT override these methods from the patch anymore.
    # (Any process already patched live keeps its good in-memory methods.)
    return
    AM = _find_class(nova_brain, "Amygdala")
    if AM is not None:
        AM.appraise = _amyg_appraise
    # Calm the LIVE amygdala instances (they were built with the hot defaults).
    host = _find_host(NB)
    if host is not None:
        sa = getattr(host, "simona_amyg", None)
        if sa is not None:
            sa.reactivity = 1.10; sa.decay = 0.85
        na = getattr(host, "nova_amyg", None)
        if na is not None:
            na.reactivity = 0.75; na.decay = 0.90


def _find_host(NB):
    """Locate the running NeuromorphicBrain instance (one-time gc scan)."""
    if NB is None:
        return None
    import gc
    for o in gc.get_objects():
        if isinstance(o, NB):
            return o
    return None


def apply_patches(nova_brain, simona_brain, shared_sem):
    SoC = _find_class(nova_brain, "StreamOfConsciousness")
    if SoC is not None:
        SoC._compose_nova = _compose_nova_uncapped
        SoC._compose_simona = _compose_simona_uncapped

    Babble = _find_class(nova_brain, "BabblingCortex")
    if Babble is not None:
        Babble.auditory_feedback = _auditory_feedback_efference

    NB = _find_class(nova_brain, "NeuromorphicBrain")
    if NB is not None:
        _install_text_presence(NB)
        _install_peer_share(NB)
        _install_neuro_fix(NB, nova_brain)

    SC = _find_class(nova_brain, "SearchCortex")
    if SC is not None:
        # Wire the running host into the search cortices so tick() can read
        # the emergent internal state (boredom/surprise/rumination) directly.
        host = _find_host(NB)
        if host is not None:
            for attr in ("nova_search", "simona_search"):
                sc = getattr(host, attr, None)
                if sc is not None:
                    sc._host = host
        SC.tick = _search_tick_emergent

    if SoC is None and Babble is None and NB is None and SC is None:
        raise RuntimeError("could not locate classes to patch")
