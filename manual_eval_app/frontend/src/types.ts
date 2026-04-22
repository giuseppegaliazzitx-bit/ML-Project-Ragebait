export interface SourceCount {
  source: string;
  labeled: number;
}

export interface TweetPayload {
  post_id: string;
  position: number;
  display_index: number;
  total: number;
  source: string;
  author_id: string;
  created_at: string;
  language: string;
  text: string;
}

export interface SessionStats {
  total_posts: number;
  completed_actions: number;
  labeled_count: number;
  skipped_count: number;
  remaining_count: number;
  next_display_index: number;
  can_undo: boolean;
  labeled_by_source: SourceCount[];
}

export interface SessionResponse {
  tweet: TweetPayload | null;
  stats: SessionStats;
}
