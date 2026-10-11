const vscode = require('vscode');

const CHAT_URL = 'https://oxalpha.com/chat';

function getWebviewContent(title = 'Ox Alpha Chat') {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: var(--vscode-editor-background, #1e1e1e);
            font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif);
        }
        #toolbar {
            height: 34px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            background-color: var(--vscode-sideBarSectionHeader-background, #252526);
            border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border, #333333);
            color: var(--vscode-foreground, #cccccc);
            font-size: 12px;
            user-select: none;
        }
        .btn-group {
            display: flex;
            gap: 6px;
        }
        button {
            background-color: var(--vscode-button-background, #0e639c);
            color: var(--vscode-button-foreground, #ffffff);
            border: none;
            padding: 3px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            transition: opacity 0.15s;
        }
        button:hover {
            background-color: var(--vscode-button-hoverBackground, #1177bb);
        }
        .btn-secondary {
            background-color: transparent;
            color: var(--vscode-foreground, #cccccc);
            border: 1px solid var(--vscode-button-border, #454545);
        }
        .btn-secondary:hover {
            background-color: var(--vscode-toolbar-hoverBackground, #383838);
        }
        iframe {
            width: 100%;
            height: calc(100% - 35px);
            border: none;
            background: #0f0f13;
        }
    </style>
</head>
<body>
    <div id="toolbar">
        <span style="font-weight: 600; display: flex; align-items: center; gap: 5px;">
            <span>🧠</span> <span>Ox Alpha Reasoning</span>
        </span>
        <div class="btn-group">
            <button class="btn-secondary" onclick="reloadFrame()" title="Reload Chat">↻ Refresh</button>
            <button class="btn-secondary" onclick="openExternal()" title="Open in External Browser">↗ Open Web</button>
        </div>
    </div>
    <iframe id="chatFrame" src="${CHAT_URL}" allow="clipboard-read; clipboard-write; microphone;"></iframe>

    <script>
        const vscode = acquireVsCodeApi();
        function reloadFrame() {
            const frame = document.getElementById('chatFrame');
            frame.src = frame.src;
        }
        function openExternal() {
            vscode.postMessage({ command: 'openExternal', url: '${CHAT_URL}' });
        }
    </script>
</body>
</html>`;
}

class OxAlphaViewProvider {
    constructor(extensionUri) {
        this._extensionUri = extensionUri;
        this._view = undefined;
    }

    resolveWebviewView(webviewView, context, _token) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = getWebviewContent('Ox Alpha Sidebar');

        webviewView.webview.onDidReceiveMessage(message => {
            if (message.command === 'openExternal') {
                vscode.env.openExternal(vscode.Uri.parse(message.url));
            }
        });
    }

    refresh() {
        if (this._view) {
            this._view.webview.html = getWebviewContent('Ox Alpha Sidebar');
        }
    }
}

function activate(context) {
    const provider = new OxAlphaViewProvider(context.extensionUri);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('oxalpha.chatView', provider, {
            webviewOptions: { retainContextWhenHidden: true }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('oxalpha.refreshChat', () => {
            provider.refresh();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('oxalpha.openChatTab', () => {
            const panel = vscode.window.createWebviewPanel(
                'oxalphaChatTab',
                'Ox Alpha Chat',
                vscode.ViewColumn.Beside,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true
                }
            );

            panel.webview.html = getWebviewContent('Ox Alpha Chat');

            panel.webview.onDidReceiveMessage(message => {
                if (message.command === 'openExternal') {
                    vscode.env.openExternal(vscode.Uri.parse(message.url));
                }
            });
        })
    );
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
