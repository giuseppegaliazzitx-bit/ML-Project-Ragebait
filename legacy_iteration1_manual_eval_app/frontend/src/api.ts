import type { SessionResponse } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json()) as { detail?: string };
      throw new Error(payload.detail || `Request failed with status ${response.status}`);
    }
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function fetchSession(): Promise<SessionResponse> {
  return request<SessionResponse>("/api/session");
}

export function submitLabel(label: 0 | 1): Promise<SessionResponse> {
  return request<SessionResponse>("/api/respond", {
    method: "POST",
    body: JSON.stringify({ action: "label", label }),
  });
}

export function submitSkip(): Promise<SessionResponse> {
  return request<SessionResponse>("/api/respond", {
    method: "POST",
    body: JSON.stringify({ action: "skip" }),
  });
}

export function undoLastAction(): Promise<SessionResponse> {
  return request<SessionResponse>("/api/undo", {
    method: "POST",
  });
}
