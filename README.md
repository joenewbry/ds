# Digital Surface

Digital Surface captures 1 screenshot each minute. It analyzes each screenshot  Claude, and stores the analyses in a searchable json format. It helps you remember what you were working on at different times and provides insights about your computer usage.

## 🎯 Goal

The primary goal of Digital Surface is to revolutionize note-taking by automatically capturing the history of what you've worked on. This enables you to:

1.  **Summarize Quickly**: Effortlessly generate summaries of your work over any period.
2.  **Recall Solutions**: Easily find solutions to problems you've tackled in the past (e.g., that elusive bug fix from six months ago).
3.  **Retroactive Documentation**: Create useful documentation for processes or tasks you've completed, even if you've forgotten the specifics.

And more to come! We're constantly looking to improve. Please email joe@digitalsurfacelabs.com with your suggestions and feedback.

## ✨ Features

- 📸 Automatically captures screenshots every 60 seconds.
- 💾 **Disk Space Considerations**: Screenshots are stored locally. The space consumed can be significant. For example, if a screenshot averages 1MB (this can vary greatly depending on screen resolution and content) and your computer is active for 8 hours a day:
  - Screenshots per day: `8 hours/day * 60 screenshots/hour = 480 screenshots/day`
  - Data per day: `480 screenshots/day * 1MB/screenshot = 480MB/day`
  - Data per week: `7 days * 480MB/day = 3360MB/week` (or 3.36 GB/week)
  This can be generalized as: `7 days * (8 hours/day * 60 screenshots/hour * S MB/screenshot) = (3360 * S) MB per week`, where `S` is the average size of a screenshot in MB. Please monitor your disk space.
- Uses Claude to analyze each screenshot with details about:
  - Active application
  - Summary of what you're doing
  - Text extraction
  - Task categorization
  - Productivity scoring
  - Workflow suggestions
- Provides a natural language query interface to search your past activities
- Supports time-based queries (e.g., "What was I working on yesterday?" or "Show me coding activities from last Tuesday")

## 🛠️ Setup

### Prerequisites

- Node.js (v16 or later)
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/digital-surface.git
   cd digital-surface
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api...
   ```

### Running the Tool

Start the tracking tool:
```
node screen-track.js
```

## 💻 Usage

### Query Interface

While the script is running, you can query your past activities directly in the terminal:

```
Query> What was I working on this morning?
Query> Show me all my coding activities from yesterday
Query> When did I last use Excel?
```

The script uses Claude to understand your natural language queries, including time-based references.

## 💡 Example Use Cases

- **Work Tracking**: "What projects did I work on Tuesday afternoon?"
- **Time Analysis**: "How much time did I spend in meetings last week?"
- **Focus Monitoring**: "When was I most productive yesterday?"
- **Activity Recall**: "What websites did I visit while researching crypto?"
- **Task Resumption**: "What was I working on before lunch?"

## 🗄️ Data Storage

Screenshots are saved in a local `screenshots` directory, and the analyses are stored in `screenhistory` as JSON files. Both directories are created automatically when you run the script.

## 🔒 Security Risks

Please be aware of the following security implications:

1.  **Sensitive Data in Screenshots**: The tool captures screenshots of your entire screen. This can inadvertently include sensitive information such as passwords, secret keys, API tokens, private messages, or confidential documents if they are visible on your screen when a screenshot is taken. These screenshots are stored locally on your disk. If your computer is compromised, or if these image files are unintentionally shared (e.g., through email, cloud sync without proper restrictions), this sensitive data could be exposed.
2.  **Data Transmission to Claude**: To analyze the content, each screenshot is sent to the Anthropic Claude API via HTTPS. While HTTPS provides a secure channel for data transmission, there's a non-zero risk of:
    *   **Interception**: Sophisticated attackers might attempt to intercept traffic (though HTTPS makes this very difficult).
    *   **Third-Party Breach**: Anthropic, like any service provider, could potentially experience a data leak or security breach, which might compromise the screenshot data sent for analysis.

It is crucial to be mindful of what is on your screen while Digital Surface is active.

## 🙈 Privacy Risks

Consider the following privacy aspects:

1.  **Continuous Monitoring**: Once started, Digital Surface runs continuously, capturing screenshots at regular intervals until you explicitly stop it. Be mindful that it records all on-screen activity during its operation, regardless of whether it's work-related or personal. We recommend pausing or stopping the tool if you are engaging in activities you do not wish to have captured.

## 🛡️ Privacy & Security

- All data is stored locally on your machine
- Your Anthropic API key is used only to communicate with the Claude API
- No data is sent to any third parties other than Anthropic for screenshot analysis

## 📦 Required Dependencies

The script uses the following npm packages:
- screenshot-desktop
- @anthropic-ai/sdk
- dotenv
- @xenova/transformers
- date-fns

## 📜 License

Digital Suraface Labs, Inc.