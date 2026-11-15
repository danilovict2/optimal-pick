"""Streamlit app: live mock draft vs bots, backtest results, weekly
start/sit -- for NFL, NBA, and Premier League (FPL Draft) leagues."""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config import LEAGUES
from draft_sim import make_pool, mc_pick, snake_order, vor_pick
from grades import draft_grade, pick_grade, simulate_reference
from league_store import list_leagues, load_league, save_league
from optimizer import best_lineup
from projections import league_engine, league_module

st.set_page_config(page_title="Optimal Pick", page_icon="🏆", layout="wide")

ACCENT = "#E23B3B"    # validated vs #0F0F10 (dataviz checks)
MUTED_BAR = "#80808A"  # de-emphasis neutral for baseline bars
INK_2 = "#9A9AA2"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600&display=swap');

h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: "Oswald", "Arial Narrow", sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600 !important;
}
[data-testid="stDataFrame"] *, [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
}
#MainMenu, [data-testid="stAppDeployButton"], footer { visibility: hidden; }

.op-brand {
    font-family: "Oswald", "Arial Narrow", sans-serif;
    font-size: 1.85rem; font-weight: 600;
    letter-spacing: 0.06em; line-height: 1.1;
    text-transform: uppercase; color: #EAEAEA;
}
.op-brand span { color: #E23B3B; }
.op-rule {
    height: 2px; width: 44px; background: #E23B3B;
    margin: 10px 0 4px 0;
}
.op-hint {
    background: #19191C; border: 1px solid #2A2A2E;
    border-radius: 4px; padding: 12px 16px; color: #C9C9CE;
}
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; }
.op-rec {
    background: #19191C; border: 1px solid #2A2A2E;
    border-left: 3px solid #E23B3B; border-radius: 4px;
    padding: 12px 18px 14px; margin-bottom: 14px;
}
.op-rec .k {
    font-size: 0.72rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #9A9AA2; margin-bottom: 2px;
}
.op-rec .v {
    font-family: "Oswald", "Arial Narrow", sans-serif;
    font-size: 1.45rem; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; color: #EAEAEA;
}
.op-rec .v .pos { color: #9A9AA2; font-size: 0.95rem; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)

def hint(text: str):
    """Neutral hint card (st.info is blue, which clashes with the theme)."""
    st.markdown(f'<div class="op-hint">{text}</div>', unsafe_allow_html=True)


@st.cache_resource(ttl="24h", show_spinner="Loading data and training model...")
def load_engine(league_key: str, teams: int):
    league = {**LEAGUES[league_key], "teams": teams}
    _, pool_df, draft_season = league_engine(league_key)
    pool = make_pool(pool_df, league)
    ref = simulate_reference(pool, n_drafts=150)
    return pool, ref, league, draft_season


# ---------- sidebar ----------
st.sidebar.markdown('<div class="op-brand">Optimal <span>Pick</span></div>'
                    '<div class="op-rule"></div>', unsafe_allow_html=True)
league_key = st.sidebar.selectbox(
    "League", list(LEAGUES), format_func=lambda k: LEAGUES[k]["label"])
CFG = LEAGUES[league_key]
MOD = league_module(league_key)
teams = int(st.sidebar.number_input("League size", 4, 20, CFG["teams"],
                                    key=f"teams_{league_key}"))
slot = st.sidebar.number_input("Your slot", 1, teams, min(5, teams),
                               key=f"slot_{league_key}")
mode = st.sidebar.radio("Mode", ["Mock draft", "Live draft helper"])
R = st.sidebar.slider("Rollouts per pick", 8, 128, 64, step=8)
seed = st.sidebar.number_input("Seed", 0, 99999, 42)

pool, ref, LEAGUE, DRAFT_SEASON = load_engine(league_key, teams)
st.sidebar.caption(f"{MOD.season_label(DRAFT_SEASON)} projections, trained "
                   f"through {MOD.season_label(DRAFT_SEASON - 1)}. Live "
                   f"mode: you enter every real pick, the sim tells you who "
                   f"to take.")
TOTAL_PICKS = LEAGUE["teams"] * LEAGUE["rounds"]
ORDER = snake_order(LEAGUE["teams"], LEAGUE["rounds"])
POS = pool["pos_order"]


def new_draft(slot: int, seed: int, live: bool, name: str):
    st.session_state.draft = {
        "slot": slot - 1,
        "league": league_key,
        "teams": LEAGUE["teams"],
        "name": name,
        "saved": False,
        "live": live,  # live helper: user records every real pick
        "rng": np.random.default_rng(seed),
        "avail": np.ones(len(pool["proj"]), dtype=bool),
        "pos_counts": np.zeros((LEAGUE["teams"], len(POS)), dtype=int),
        "rosters": [[] for _ in range(LEAGUE["teams"])],
        "picks": [],  # (round, team, player_idx)
        "pick_no": 0,
        "rec": None,  # cached MC recommendation for the current pick
    }


def apply_pick(d: dict, i: int):
    r, t = ORDER[d["pick_no"]]
    d["avail"][i] = False
    d["pos_counts"][t][pool["pos_codes"][i]] += 1
    d["rosters"][t].append(int(i))
    d["picks"].append((r, t, int(i)))
    d["pick_no"] += 1
    d["rec"] = None


def undo_pick(d: dict):
    if d["picks"]:
        r, t, i = d["picks"].pop()
        d["avail"][i] = True
        d["pos_counts"][t][pool["pos_codes"][i]] -= 1
        d["rosters"][t].pop()
        d["pick_no"] -= 1
        d["rec"] = None


def advance_bots(d: dict):
    while d["pick_no"] < TOTAL_PICKS and ORDER[d["pick_no"]][1] != d["slot"]:
        r, t = ORDER[d["pick_no"]]
        apply_pick(d, vor_pick(pool, d["rng"], d["avail"], d["pos_counts"][t], r))


def player_select(d: dict, label: str, key: str):
    """Searchable dropdown over available players, best VOR first."""
    avail_idx = np.flatnonzero(d["avail"])
    avail_idx = avail_idx[np.argsort(pool["vor"][avail_idx])[::-1]]
    return st.selectbox(
        label, avail_idx, key=key,
        format_func=lambda i: f"{pool['names'][i]} "
        f"({POS[pool['pos_codes'][i]]}, {pool['proj'][i]:.0f})")


def board_df(d: dict) -> pd.DataFrame:
    other = "Team" if d["live"] else "Bot"
    rows = [{"Pick": n + 1, "Round": r + 1,
             "Team": "You" if t == d["slot"] else f"{other} {t + 1}",
             "Player": pool["names"][i], "Pos": POS[pool["pos_codes"][i]],
             "Proj": round(float(pool["proj"][i]), 1)}
            for n, (r, t, i) in enumerate(d["picks"])]
    return pd.DataFrame(rows)


def roster_df(roster: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "name": [pool["names"][i] for i in roster],
        "position": [POS[pool["pos_codes"][i]] for i in roster],
        "proj": pool["proj"][list(roster)],
    })


def roster_records(roster: list[int]) -> list[dict]:
    df = pool["df"]
    return [{"player_id": str(df.loc[i, "player_id"]),
             "name": pool["names"][i],
             "position": POS[pool["pos_codes"][i]],
             "proj": round(float(pool["proj"][i]), 1)} for i in roster]


league_name = st.sidebar.text_input("League name", "My League")
if st.sidebar.button("Start new draft", type="primary"):
    new_draft(slot, seed, live=mode.startswith("Live"), name=league_name)

tab_draft, tab_team, tab_back = st.tabs(["Draft", "My Team", "Backtest"])

# ---------- Draft ----------
with tab_draft:
    if "draft" not in st.session_state:
        hint("Pick a slot in the sidebar and start a draft.")
    elif (st.session_state.draft.get("league") != league_key
          or st.session_state.draft["teams"] != LEAGUE["teams"]):
        st.warning("League or league size changed mid-draft. Switch back "
                   "or start a new draft.")
    else:
        d = st.session_state.draft
        if not d["live"]:
            advance_bots(d)
        left, right = st.columns([3, 2])

        with right:
            st.subheader("Draft board")
            b = board_df(d)
            if len(b):
                st.dataframe(b.iloc[::-1], height=420, hide_index=True)

        with left:
            if d["pick_no"] >= TOTAL_PICKS:
                st.subheader("Draft complete")
                if not d["saved"]:
                    d["name"] = save_league(
                        d["name"], LEAGUE, DRAFT_SEASON, d["slot"],
                        [roster_records(ro) for ro in d["rosters"]], d["live"],
                        league_key)
                    d["saved"] = True
                st.caption(f"Saved as \"{d['name']}\". Weekly lineups are in "
                           f"My Team.")
                total, lineup = best_lineup(roster_df(d["rosters"][d["slot"]]), LEAGUE)
                st.markdown(f"**Best lineup: {total:.0f} projected pts**")
                st.dataframe(lineup[["slot", "name", "position", "proj"]].round(1),
                             hide_index=True)
                rows = []
                for n, (r, t, i) in enumerate(d["picks"]):
                    if t == d["slot"]:
                        letter, pctl, vs = pick_grade(pool, i, n, ref)
                        rows.append({"Round": r + 1, "Player": pool["names"][i],
                                     "Pos": POS[pool["pos_codes"][i]],
                                     "Grade": letter, "Pctl": round(pctl),
                                     "vs pick avg": round(vs, 1)})
                st.markdown("**Pick grades**")
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                letter, pctl, val = draft_grade(pool, d["rosters"][d["slot"]],
                                                d["slot"], ref)
                st.metric("Draft grade", letter)
                st.caption(f"Roster value {val:.0f}, beats {pctl:.0f}% of sim "
                           f"drafts from slot {d['slot'] + 1}.")
            elif ORDER[d["pick_no"]][1] == d["slot"]:
                r_now, _ = ORDER[d["pick_no"]]
                st.subheader(f"Pick {d['pick_no'] + 1} (round {r_now + 1}): "
                             f"you're up")
                if d["rec"] is None:
                    with st.spinner("Simulating..."):
                        _, cands = mc_pick(pool, d["rng"], d["avail"],
                                           d["pos_counts"], d["rosters"][d["slot"]],
                                           d["slot"], ORDER, d["pick_no"], K=8, R=R)
                    d["rec"] = cands
                cands = d["rec"]
                best = cands[0]
                st.markdown(
                    f'<div class="op-rec"><div class="k">Recommended pick'
                    f'</div><div class="v">{best["name"]}'
                    f'<span class="pos">{best["position"]}</span></div></div>',
                    unsafe_allow_html=True)
                why = pd.DataFrame(cands)[["name", "position", "proj", "mean_value"]]
                why.columns = ["Player", "Pos", "Proj", "EV"]
                st.dataframe(why.round(1), hide_index=True)
                st.caption(f"EV = avg final roster value, {R} sims each.")

                choice = player_select(d, "Player", "my_pick")
                if st.button("Draft", type="primary"):
                    apply_pick(d, int(choice))
                    st.rerun()
                if d["live"] and st.button("Undo"):
                    undo_pick(d)
                    st.rerun()
                if d["rosters"][d["slot"]]:
                    st.markdown("**Your roster**")
                    st.dataframe(roster_df(d["rosters"][d["slot"]]).round(1),
                                 hide_index=True)
            else:
                # live helper: user enters what another team actually picked
                r_now, t_now = ORDER[d["pick_no"]]
                st.subheader(f"Pick {d['pick_no'] + 1} (round {r_now + 1}): "
                             f"Team {t_now + 1}")
                choice = player_select(d, "They took", "opp_pick")
                col1, col2 = st.columns(2)
                if col1.button("Record", type="primary"):
                    apply_pick(d, int(choice))
                    st.rerun()
                if col2.button("Undo"):
                    undo_pick(d)
                    st.rerun()
                if d["rosters"][d["slot"]]:
                    st.markdown("**Your roster**")
                    st.dataframe(roster_df(d["rosters"][d["slot"]]).round(1),
                                 hide_index=True)

# ---------- Backtest ----------
with tab_back:
    csv = Path(f"backtest_results_{league_key}.csv")
    bt_label = MOD.season_label(CFG["backtest_season"])
    if not csv.exists():
        hint(f"Run `python backtest.py {league_key}` first to generate the "
             f"{CFG['label']} backtest.")
    else:
        df = pd.read_csv(csv)
        names = {"mc": "Monte Carlo (ours)", "best_available": "Best available",
                 "vor": "VOR bot"}
        df["strategy"] = df["strategy"].map(names)
        summary = df.groupby("strategy").agg(
            **{"Actual pts": ("actual_points", "mean"),
               "Avg finish": ("rank", "mean"),
               "% 1st": ("rank", lambda r: 100 * (r == 1).mean()),
               "% top 3": ("rank", lambda r: 100 * (r <= 3).mean())}
        ).round(2).sort_values("Avg finish")

        mc = summary.loc[names["mc"]]
        n = df.groupby("strategy").size().iloc[0]
        bt_teams = int(df["slot"].max())
        st.subheader(f"{bt_label} backtest")
        st.write(f"{n} drafts per strategy, {bt_teams}-team leagues, every "
                 f"roster scored week by week with its best lineup from what "
                 f"players actually did in {bt_label} (absences count). Our "
                 f"bot finished top 3 {mc['% top 3']:.0f}% of the time, "
                 f"average finish {mc['Avg finish']:.1f} of {bt_teams}.")
        st.dataframe(summary)

        chart_df = summary.reset_index()[["strategy", "Avg finish"]]
        chart_df["ours"] = chart_df["strategy"] == names["mc"]
        base = alt.Chart(chart_df).encode(
            x=alt.X("Avg finish:Q", scale=alt.Scale(domain=[0, bt_teams]),
                    axis=alt.Axis(values=list(range(0, bt_teams + 1, 2)),
                                  grid=False),
                    title="Avg finish (lower is better)"),
            y=alt.Y("strategy:N", sort="x", title=None),
            tooltip=["strategy", "Avg finish"],
        )
        bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
            color=alt.condition(alt.datum.ours, alt.value(ACCENT),
                                alt.value(MUTED_BAR)))
        labels = base.mark_text(align="left", dx=6, color="#EAEAEA").encode(
            text=alt.Text("Avg finish:Q", format=".1f"))
        st.altair_chart(
            (bars + labels)
            .configure_view(strokeWidth=0)
            .configure_axis(domain=False, tickColor="transparent",
                            gridColor="#232327", labelColor=INK_2,
                            titleColor=INK_2),
            width="stretch")

# ---------- My Team ----------
with tab_team:
    form, period, form_note = MOD.current_form()

    def weekly_roster(team: list[dict]) -> pd.DataFrame:
        """Attach current form + availability status to a saved roster."""
        rdf = pd.DataFrame(team)[["player_id", "name", "position"]]
        rdf = rdf.merge(form[["player_id", "proj", "status"]],
                        on="player_id", how="left")
        rdf["status"] = rdf["status"].fillna("no games this season")
        rdf["proj"] = rdf["proj"].fillna(0.0)
        return rdf

    saved = list_leagues(league_key)
    if not saved:
        hint("No leagues yet. Draft one first.")
    else:
        pick_league = st.selectbox("League", saved)
        lg = load_league(pick_league)
        lg_cfg = {**LEAGUES[lg.get("league", "nfl")],
                  "teams": lg["teams"], "rounds": lg["rounds"]}
        mine = lg["rosters"][lg["my_slot"]]
        st.caption(f"Drafted for {MOD.season_label(lg['draft_season'])} on "
                   f"{lg['created']}, slot {lg['my_slot'] + 1} of "
                   f"{lg['teams']}. {form_note} Reruns on every data "
                   f"refresh.")

        left, right = st.columns([3, 2])
        with left:
            rdf = weekly_roster(mine)
            total, lineup = best_lineup(rdf, lg_cfg)
            st.markdown(f"**{period} lineup ({total:.1f} pts)**")
            forced = lineup[lineup["proj"] <= 0]
            if len(forced):
                st.warning("Nobody available for: " + "; ".join(
                    f"{r['slot']} ({r['name']}: {r['status'] or 'no points'})"
                    for _, r in forced.iterrows()) + ". Waiver pickup needed.")
            st.dataframe(
                lineup[["slot", "name", "position", "proj", "status"]].round(1),
                hide_index=True)
            bench = rdf[~rdf["name"].isin(lineup["name"])]
            if len(bench):
                st.markdown("**Sit**")
                st.dataframe(
                    bench[["name", "position", "proj", "status"]].round(1)
                    .sort_values("proj", ascending=False), hide_index=True)

        with right:
            st.markdown(f"**{period} projected scores**")
            rows = []
            for t, team in enumerate(lg["rosters"]):
                val, _ = best_lineup(weekly_roster(team), lg_cfg)
                rows.append({"Team": "You" if t == lg["my_slot"]
                             else f"Team {t + 1}", "Proj": round(val, 1)})
            board = pd.DataFrame(rows).sort_values("Proj", ascending=False)
            board.insert(0, "Rank", range(1, len(board) + 1))
            st.dataframe(board, hide_index=True, height=460)

    with st.expander("Check a roster that isn't saved here"):
        fi = form.set_index("player_id")
        picked = st.multiselect(
            "Players", form.sort_values("form", ascending=False)["player_id"],
            format_func=lambda pid: (lambda r: f"{r['name']} "
                                     f"({r['position']}, {r['form']:.1f}/gm)")(
                                         fi.loc[pid]))
        if picked:
            rdf = form[form["player_id"].isin(picked)]
            total, lineup = best_lineup(rdf, LEAGUE)
            st.markdown(f"**Start these ({total:.1f} pts):**")
            st.dataframe(
                lineup[["slot", "name", "position", "proj", "status"]].round(1),
                hide_index=True)
