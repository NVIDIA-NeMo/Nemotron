/**
 * termynal-init.js
 * Sphinx/MyST integration for Termynal terminal animations.
 *
 * Parses code blocks inside .termy containers and converts them
 * to animated terminal widgets.
 */

'use strict';

function setupTermynal() {
    const progressLiteralStart = "---> 100%";
    const promptLiteralStart = "$ ";
    const customPromptLiteralStart = "# ";
    const termynalActivateClass = "termy";
    let termynals = [];

    function createTermynals() {
        // Find all .termy containers with code blocks inside
        // Sphinx/MyST generates: .termy > .highlight > pre
        document
            .querySelectorAll(`.${termynalActivateClass}`)
            .forEach(container => {
                // Find the code content - try different selectors for Sphinx HTML structure
                const codeNode = container.querySelector('.highlight pre') ||
                                 container.querySelector('pre') ||
                                 container.querySelector('code');

                if (!codeNode) {
                    console.warn('Termynal: No code block found in .termy container');
                    return;
                }

                const text = codeNode.textContent;
                const lines = text.split("\n");
                const useLines = [];
                let buffer = [];

                function saveBuffer() {
                    if (buffer.length) {
                        let isBlankSpace = true;
                        buffer.forEach(line => {
                            if (line) {
                                isBlankSpace = false;
                            }
                        });
                        let dataValue = {};
                        if (isBlankSpace) {
                            dataValue["delay"] = 0;
                        }
                        if (buffer[buffer.length - 1] === "") {
                            // A last single <br> won't have effect
                            // so put an additional one
                            buffer.push("");
                        }
                        const bufferValue = buffer.join("<br>");
                        dataValue["value"] = bufferValue;
                        useLines.push(dataValue);
                        buffer = [];
                    }
                }

                for (let line of lines) {
                    if (line === progressLiteralStart) {
                        saveBuffer();
                        useLines.push({
                            type: "progress"
                        });
                    } else if (line.startsWith(promptLiteralStart)) {
                        saveBuffer();
                        const value = line.replace(promptLiteralStart, "").trimEnd();
                        useLines.push({
                            type: "input",
                            value: value
                        });
                    } else if (line.startsWith("// ")) {
                        saveBuffer();
                        const value = "💬 " + line.replace("// ", "").trimEnd();
                        useLines.push({
                            value: value,
                            class: "termynal-comment",
                            delay: 0
                        });
                    } else if (line.startsWith(customPromptLiteralStart)) {
                        saveBuffer();
                        const promptStart = line.indexOf(promptLiteralStart);
                        if (promptStart === -1) {
                            console.error("Custom prompt found but no end delimiter", line)
                            buffer.push(line);
                            continue;
                        }
                        const prompt = line.slice(0, promptStart).replace(customPromptLiteralStart, "")
                        let value = line.slice(promptStart + promptLiteralStart.length);
                        useLines.push({
                            type: "input",
                            value: value,
                            prompt: prompt
                        });
                    } else {
                        buffer.push(line);
                    }
                }
                saveBuffer();

                // Create new container for termynal
                const div = document.createElement("div");
                container.innerHTML = '';
                container.appendChild(div);

                const termynal = new Termynal(div, {
                    lineData: useLines,
                    noInit: true,
                    lineDelay: 500
                });
                termynals.push(termynal);
            });
    }

    function loadVisibleTermynals() {
        termynals = termynals.filter(termynal => {
            if (termynal.container.getBoundingClientRect().top - innerHeight <= 0) {
                termynal.init();
                return false;
            }
            return true;
        });
    }

    window.addEventListener("scroll", loadVisibleTermynals);
    createTermynals();
    loadVisibleTermynals();
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupTermynal);
} else {
    setupTermynal();
}
