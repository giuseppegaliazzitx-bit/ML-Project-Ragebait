import { useEffect, useState } from "react";

import { fetchSession, submitLabel, submitSkip, undoLastAction } from "./api";
import type { SessionResponse } from "./types";

type ActionHandler = () => Promise<SessionResponse>;

function App() {
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [showInfo, setShowInfo] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadSession();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || submitting) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "1") {
        event.preventDefault();
        void handleAction(() => submitLabel(0));
      } else if (key === "2") {
        event.preventDefault();
        void handleAction(() => submitLabel(1));
      } else if (key === "s") {
        event.preventDefault();
        void handleAction(submitSkip);
      } else if (key === "z") {
        event.preventDefault();
        void handleAction(undoLastAction);
      } else if (key === "i") {
        event.preventDefault();
        setShowInfo((current) => !current);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [submitting]);

  async function loadSession() {
    setLoading(true);
    setError(null);
    try {
      setSession(await fetchSession());
    } catch (loadError) {
      setError(getMessage(loadError));
    } finally {
      setLoading(false);
    }
  }

  async function handleAction(action: ActionHandler) {
    setSubmitting(true);
    setError(null);
    try {
      setSession(await action());
    } catch (actionError) {
      setError(getMessage(actionError));
    } finally {
      setSubmitting(false);
    }
  }

  const stats = session?.stats;
  const tweet = session?.tweet;
  const progressRatio =
    stats && stats.total_posts > 0
      ? Math.min(stats.completed_actions / stats.total_posts, 1)
      : 0;

  return (
    <div className="shell">
      <div className="backdrop" />
      <main className="layout">
        <section className="review-panel">
          <header className="panel-card header-card">
            <div>
              <p className="eyebrow">Manual Eval</p>
              <h1>Tweet Ragebait Review</h1>
              <p className="subcopy">
                Single-tweet labeling with persistent save, balanced source rotation,
                and one-step undo.
              </p>
            </div>
            <div className="header-actions">
              <button
                className="ghost-button"
                type="button"
                onClick={() => setShowInfo((current) => !current)}
              >
                {showInfo ? "Hide Info" : "Show Info"}
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => void handleAction(undoLastAction)}
                disabled={!stats?.can_undo || submitting}
              >
                Undo
              </button>
            </div>
          </header>

          <section className="panel-card progress-card">
            <div className="progress-meta">
              <span>Progress</span>
              <span>
                {stats?.completed_actions ?? 0} / {stats?.total_posts ?? 0}
              </span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progressRatio * 100}%` }}
              />
            </div>
          </section>

          {error ? (
            <section className="panel-card error-card">
              <strong>Request failed.</strong>
              <p>{error}</p>
              <button className="ghost-button" type="button" onClick={() => void loadSession()}>
                Retry
              </button>
            </section>
          ) : null}

          {loading ? (
            <section className="panel-card tweet-card loading-card">
              <div className="spinner" />
              <p>Loading queue…</p>
            </section>
          ) : tweet ? (
            <section className="panel-card tweet-card">
              <div className="tweet-meta">
                <span className="badge">Tweet {tweet.display_index}</span>
                <span className="badge muted">{tweet.source}</span>
                <span className="badge muted">
                  Author bucket: {tweet.author_id}
                </span>
              </div>
              <article className="tweet-text">{tweet.text}</article>
              <footer className="tweet-footer">
                <div className="hint-grid">
                  <span>`1` not ragebait</span>
                  <span>`2` ragebait</span>
                  <span>`S` skip</span>
                  <span>`Z` undo</span>
                </div>
                <div className="action-grid">
                  <button
                    className="action-button calm"
                    type="button"
                    onClick={() => void handleAction(() => submitLabel(0))}
                    disabled={submitting}
                  >
                    0 · Not Ragebait
                  </button>
                  <button
                    className="action-button hot"
                    type="button"
                    onClick={() => void handleAction(() => submitLabel(1))}
                    disabled={submitting}
                  >
                    1 · Ragebait
                  </button>
                  <button
                    className="action-button neutral"
                    type="button"
                    onClick={() => void handleAction(submitSkip)}
                    disabled={submitting}
                  >
                    Skip
                  </button>
                </div>
              </footer>
            </section>
          ) : (
            <section className="panel-card tweet-card complete-card">
              <p className="eyebrow">Queue Complete</p>
              <h2>Everything in the current queue has an action.</h2>
              <p>
                Use undo if you want to revisit the most recent decisions. Labeled
                rows are already written to <code>data/labeled/manual_eval.csv</code>.
              </p>
            </section>
          )}
        </section>

        {showInfo ? (
          <aside className="info-panel">
            <section className="panel-card info-card">
              <p className="eyebrow">Info</p>
              <h2>Manual evaluation status</h2>
              <div className="stat-list">
                <div className="stat-row">
                  <span>Manually evaluated</span>
                  <strong>{stats?.labeled_count ?? 0}</strong>
                </div>
                <div className="stat-row">
                  <span>Skipped</span>
                  <strong>{stats?.skipped_count ?? 0}</strong>
                </div>
                <div className="stat-row">
                  <span>Remaining in queue</span>
                  <strong>{stats?.remaining_count ?? 0}</strong>
                </div>
                <div className="stat-row">
                  <span>Saved output</span>
                  <strong>data/labeled/manual_eval.csv</strong>
                </div>
              </div>
            </section>

            <section className="panel-card info-card">
              <p className="eyebrow">Source Mix</p>
              <h2>Labeled by source</h2>
              {stats?.labeled_by_source.length ? (
                <div className="source-list">
                  {stats.labeled_by_source.map((item) => (
                    <div className="source-row" key={item.source}>
                      <span>{item.source}</span>
                      <strong>{item.labeled}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-copy">No saved labels yet.</p>
              )}
            </section>

            <section className="panel-card info-card">
              <p className="eyebrow">Shortcuts</p>
              <h2>Faster labeling</h2>
              <div className="shortcut-list">
                <div className="shortcut-row">
                  <kbd>1</kbd>
                  <span>Mark as not ragebait</span>
                </div>
                <div className="shortcut-row">
                  <kbd>2</kbd>
                  <span>Mark as ragebait</span>
                </div>
                <div className="shortcut-row">
                  <kbd>S</kbd>
                  <span>Skip current tweet</span>
                </div>
                <div className="shortcut-row">
                  <kbd>Z</kbd>
                  <span>Undo last action</span>
                </div>
                <div className="shortcut-row">
                  <kbd>I</kbd>
                  <span>Toggle this panel</span>
                </div>
              </div>
            </section>
          </aside>
        ) : null}
      </main>
    </div>
  );
}

function getMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unknown error";
}

export default App;
