# CommentPulse Privacy Policy

Last updated: October 14, 2026

CommentPulse is a browser extension that analyzes comments displayed on the active YouTube video page when you open the extension.

## Data processed

When you use the extension, it reads the visible comment text from the current YouTube page and creates timestamps for those comments. It sends the comment text and timestamps over HTTPS to the CommentPulse API at https://commentpulse-api.onrender.com to generate sentiment and related analytics. The extension does not collect YouTube passwords, Google account credentials, browsing history, or commenter profile information.

The extension stores only your selected API base URL in the browser's local extension storage. It does not send analytics or telemetry about extension usage.

## Retention and sharing

Comment text is processed in memory by the API and is not intentionally written to a persistent database or used to train models. Some analytics requests run as temporary background jobs; job status and results are held in service memory for up to 30 minutes and are lost if the service restarts. The API records operational request metadata for service health and troubleshooting, but does not intentionally log comment text.

The API is hosted by Render. Network and infrastructure providers may process connection and operational data as part of delivering the service. CommentPulse does not sell comment data or share it for advertising.

## Your choices

Use the extension only when you want to analyze the comments currently visible on a YouTube page. You can remove the extension or clear its stored API URL through the extension's API settings. You can also use a different API endpoint that you control.

## Contact

For questions or privacy requests, contact the project maintainers through [the CommentPulse GitHub repository](https://github.com/D-393Patel/CommentPulse).
