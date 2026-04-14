# Rage-Bait Annotation Guidelines

## Goal

Label whether a post is intentionally crafted to provoke anger, pile-ons, or hostile engagement.

- `1 = Rage-Bait`
- `0 = Not Rage-Bait`

## Positive Label: Rage-Bait

Assign `1` when the text is primarily trying to trigger outrage or retaliatory engagement rather than communicate sincere information or emotion.

Common signals:

- deliberately inflammatory phrasing aimed at a broad audience
- taunts such as "cry harder", "stay mad", or "prove me right in the replies"
- blanket insults toward groups intended to spark a backlash
- obvious provocation framed as a challenge to react
- statements that appear optimized for quote-tweets, dogpiles, or rage clicks

Examples:

- "if this makes you furious, congratulations, you played exactly the role i expected"
- "everyone who disagrees with me is stupid and i hope the replies prove it"

## Negative Label: Not Rage-Bait

Assign `0` when the post is not intentionally engineered to provoke rage, even if the content is emotional, political, critical, or controversial.

This includes:

- genuine outrage about a real event
- news updates, commentary, or reporting
- neutral conversation
- sarcasm without a clear engagement-bait goal
- personal frustration not aimed at inciting mass conflict

Examples:

- "i am genuinely angry about the service outage and the lack of communication"
- "new city budget proposal released today; here are the main changes"

## Distinguishing Rage-Bait From Genuine Outrage

Label as `1` only when the post appears strategically provocative.

Stronger evidence for rage-bait:

- invites angry replies or mocks expected reactions
- contains broad, inflammatory overgeneralizations with little substance
- looks intentionally low-evidence but high-conflict
- reads like a performance for engagement rather than sincere reaction

Stronger evidence for genuine outrage:

- references a concrete event, harm, or grievance
- expresses anger without taunting the audience
- focuses on accountability, facts, or lived experience

## Ambiguous Cases

If a post is ambiguous:

1. Prefer `0` unless provocation appears intentional.
2. Add a short note in the `notes` column.
3. Route uncertain items for second review.

## Exclusions

Exclude or flag posts that cannot be reliably interpreted:

- media-only posts with no readable text
- empty submissions
- machine-generated noise
- unsupported languages if the training corpus is English-only

## Quality Controls

- Use two annotators for edge cases or high-impact samples.
- Review class balance periodically so the dataset does not collapse into mostly neutral content.
- Track inter-annotator agreement and refresh examples when drift appears.
- Sample difficult negatives such as sincere political outrage to prevent the model from learning "anger = rage-bait".

