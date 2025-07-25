document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const queryForm = document.getElementById('queryForm');
    const questionInput = document.getElementById('questionInput');
    const chatContainer = document.getElementById('chatContainer');
    const emptyState = document.getElementById('emptyState');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    const loadingStatus = document.getElementById('loadingStatus');
    const newChatBtn = document.getElementById('newChatBtn');

    const mainContent = document.getElementById('mainContent');
    const submitButton = document.getElementById('submitButton');

    // State Management
    let currentQueryId = null;
    let currentChart = null;
    let pollingInterval = null;
    const MAX_POLL_ATTEMPTS = 30;
    const POLL_INTERVAL = 1000;
    let isTyping = false;

    // Initialize
    initEventListeners();

    function initEventListeners() {
        queryForm.addEventListener('submit', handleQuerySubmit);
        questionInput.addEventListener('keydown', handleInputKeydown);
        
        // Add click listeners for example queries
        document.querySelectorAll('.example-query').forEach(item => {
            item.addEventListener('click', function() {
                const queryText = this.textContent.replace(/^"\s*|\s*"$/g, '').replace(/^[^"]+"|"[^"]+$/g, '').trim();
                questionInput.value = queryText;
                questionInput.focus();
            });
        });
    }

    
    

    function handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            queryForm.dispatchEvent(new Event('submit'));
        }
    }

    async function handleQuerySubmit(e) {
        e.preventDefault();
        const question = questionInput.value.trim();
        
        if (!validateInput(question)) return;
        
        // Add user message to chat
        addMessage(question, 'user');
        questionInput.value = '';
        
        // Show loading indicator
        showTypingIndicator();
        
        // Disable submit button during processing
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="bi bi-hourglass"></i>';
        
        try {
            const response = await submitQuery(question);
            currentQueryId = response.query_id;
            startResultPolling(currentQueryId);
        } catch (error) {
            handleQueryError(error);
            resetSubmitButton();
        }
    }

    function resetSubmitButton() {
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="bi bi-send-fill"></i>';
        anime.remove(submitButton);
    }

    function validateInput(question) {
        if (!question) {
            anime({
                targets: questionInput,
                translateX: [ -10, 10, -10, 0 ],
                duration: 400,
                easing: 'easeInOutSine'
            });
            return false;
        }
        return true;
    }

    function addMessage(content, sender) {
    if (emptyState.style.display !== 'none') {
        anime({
            targets: emptyState,
            opacity: [1, 0],
            scale: [1, 0.9],
            duration: 300,
            easing: 'easeInOutSine',
            complete: () => {
                emptyState.style.display = 'none';
            }
        });
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const now = new Date();
    const timestamp = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `

        <div class="message-content">
            ${sender === 'user' ? content : ''}
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    
    anime({
        targets: messageDiv,
        opacity: [0, 1],
        translateY: [20, 0],
        duration: 400,
        easing: 'easeOutExpo'
    });
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    if (sender === 'bot') {
        return messageDiv.querySelector('.message-content');
    }
}
    function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        
        <div class="message-meta">
        <div class="typing-content">
            <div class="typing-text">Analyzing data</div>
            <div class="typing-animation">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    chatContainer.appendChild(typingDiv);
    
    anime({
        targets: typingDiv,
        opacity: [0, 1],
        translateY: [20, 0],
        duration: 300,
        easing: 'easeOutExpo'
    });
    
    const typingText = typingDiv.querySelector('.typing-text');
    let dots = 0;
    const dotInterval = setInterval(() => {
        dots = (dots + 1) % 4;
        typingText.textContent = 'Analyzing data' + '.'.repeat(dots);
    }, 500);
    
    typingDiv.dataset.interval = dotInterval;
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

    function typeResponse(element, text) {
    if (isTyping) return;
    isTyping = true;
    
    // First, parse the text for markdown-like formatting
    const formattedText = formatResponseText(text);
    
    let i = 0;
    const speed = 20;
    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    element.appendChild(cursor);
    
    function typeWriter() {
        if (i < formattedText.length) {
            element.insertBefore(document.createTextNode(formattedText.charAt(i)), cursor);
            i++;
            setTimeout(typeWriter, speed);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } else {
            cursor.remove();
            isTyping = false;
            // After typing is done, process any special formatting
            processResponseFormatting(element);
        }
    }
    
    typeWriter();
}

    function formatResponseText(text) {
        // Simple markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markers (we'll add CSS later)
            .replace(/\*(.*?)\*/g, '$1')    // Remove italic markers
            .replace(/`(.*?)`/g, '$1');     // Remove code markers
    }

    function processResponseFormatting(element) {
        // Add HTML formatting based on markdown-like syntax
        const text = element.innerHTML;
        
        // Process bold (**text**)
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Process italic (*text*)
        formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Process code (`code`)
        formattedText = formattedText.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Process paragraphs (double newline)
        formattedText = formattedText.replace(/\n\n/g, '</p><p>');
        
        // Process line breaks (single newline)
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        // Wrap in paragraph tags if needed
        if (!formattedText.startsWith('<p>')) {
            formattedText = '<p>' + formattedText + '</p>';
        }
        
        element.innerHTML = formattedText;
        
        // Add classes to code blocks
        const codeBlocks = element.querySelectorAll('code');
        codeBlocks.forEach(code => {
            const pre = document.createElement('pre');
            pre.className = 'code-block';
            code.parentNode.replaceChild(pre, code);
            pre.appendChild(code);
        });
    }



    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            clearInterval(typingIndicator.dataset.interval);
            
            anime({
                targets: typingIndicator,
                opacity: [1, 0],
                translateY: [0, 20],
                duration: 200,
                easing: 'easeInExpo',
                complete: function() {
                    typingIndicator.remove();
                }
            });
        }
    }

    async function submitQuery(question) {
        loadingModal.show();
        loadingStatus.textContent = "Analyzing your question...";
        
        anime({
            targets: '#loadingModal .modal-content',
            opacity: [0, 1],
            scale: [0.9, 1],
            duration: 300,
            easing: 'easeOutExpo'
        });
        
        const progressSteps = [
            "Parsing natural language query",
            "Connecting to data sources",
            "classifying relevant data",
            "Executing database query",
            "Analyzing results",
            "Preparing visualization"
        ];
        
        let step = 0;
        const progressInterval = setInterval(() => {
            if (step < progressSteps.length - 1) {
                step++;
                loadingStatus.textContent = progressSteps[step];
            }
        }, 1000);
        
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({ question })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            clearInterval(progressInterval);
            return await response.json();
        } catch (error) {
            clearInterval(progressInterval);
            throw error;
        }
    }
    
    function startResultPolling(queryId) {
        let attempts = 0;
        
        pollingInterval = setInterval(async () => {
            attempts++;
            
            try {
                const result = await fetchQueryResult(queryId);
                
                if (result.status === 'completed') {
                    handleSuccessfulQuery(result);
                    clearInterval(pollingInterval);
                } else if (result.status === 'failed') {
                    handleFailedQuery();
                    clearInterval(pollingInterval);
                } else {
                    updatePollingStatus(result, attempts);
                }
                
                if (attempts >= MAX_POLL_ATTEMPTS) {
                    handlePollingTimeout();
                    clearInterval(pollingInterval);
                }
            } catch (error) {
                handlePollingError(error);
                clearInterval(pollingInterval);
            }
        }, POLL_INTERVAL);
    }

    async function fetchQueryResult(queryId) {
        const response = await fetch(`/api/query/${queryId}`);
        
        if (response.status === 404) {
            throw new Error('Result not ready');
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    function handleSuccessfulQuery(result) {
        loadingModal.hide();
        removeTypingIndicator();
        resetSubmitButton();
        
        const responseContentDiv = addMessage('', 'bot');
        
        if (result.visualization_type) {
            const chartContainer = document.createElement('div');
            chartContainer.className = 'chart-container mt-3';
            chartContainer.id = 'chartCanvas';
            responseContentDiv.parentNode.insertBefore(chartContainer, responseContentDiv.nextSibling);
        }
        
        const actionButtons = document.createElement('div');
        actionButtons.className = 'action-buttons';
        
        if (result.data && result.data.length > 0) {
            const viewDataBtn = document.createElement('button');
            viewDataBtn.className = 'btn btn-action btn-view-data';
            viewDataBtn.innerHTML = '<i class="bi bi-table"></i> View Data';
            viewDataBtn.onclick = () => toggleDataView(result.columns, result.data);
            actionButtons.appendChild(viewDataBtn);
            
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'btn btn-action btn-download';
            downloadBtn.innerHTML = '<i class="bi bi-download"></i> Download CSV';
            downloadBtn.onclick = () => downloadResults(currentQueryId);
            actionButtons.appendChild(downloadBtn);
        }
        
        if (result.sql_query) {
            const viewQueryBtn = document.createElement('button');
            viewQueryBtn.className = 'btn btn-action btn-view-query';
            viewQueryBtn.innerHTML = '<i class="bi bi-code-square"></i> View Query';
            viewQueryBtn.onclick = () => toggleQueryView(result.sql_query);
            actionButtons.appendChild(viewQueryBtn);
            
            const copyQueryBtn = document.createElement('button');
            copyQueryBtn.className = 'btn btn-action';
            copyQueryBtn.innerHTML = '<i class="bi bi-clipboard"></i> Copy Query';
            copyQueryBtn.onclick = () => copyToClipboard(result.sql_query);
            actionButtons.appendChild(copyQueryBtn);
        }
        
        responseContentDiv.parentNode.appendChild(actionButtons);
        
        if (result.data && result.data.length > 0) {
            const dataTable = createDataTable(result.columns, result.data);
            dataTable.className += ' hidden-content mt-3';
            dataTable.id = 'dataTable';
            responseContentDiv.parentNode.appendChild(dataTable);
        }
        
        if (result.sql_query) {
            const queryDiv = document.createElement('pre');
            queryDiv.className = 'hidden-content bg-light p-3 rounded mt-3';
            queryDiv.id = 'queryContent';
            queryDiv.textContent = result.sql_query;
            responseContentDiv.parentNode.appendChild(queryDiv);
        }
        
        typeResponse(responseContentDiv, result.answer);
        
        if (result.visualization_type) {
            setTimeout(() => {
                createVisualization(result.visualization_type, result.columns, result.data);
            }, 500);
        }
    }

    function typeResponse(element, text) {
        if (isTyping) return;
        isTyping = true;
        
        let i = 0;
        const speed = 20;
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        element.appendChild(cursor);
        
        function typeWriter() {
            if (i < text.length) {
                element.insertBefore(document.createTextNode(text.charAt(i)), cursor);
                i++;
                setTimeout(typeWriter, speed);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            } else {
                cursor.remove();
                isTyping = false;
            }
        }
        
        typeWriter();
    }

    function createVisualization(type, columns, data) {
        const chartContainer = document.getElementById('chartCanvas');
        if (!chartContainer) return;
        
        if (currentChart) {
            currentChart.destroy();
        }
        
        chartContainer.classList.add('loading');
        const canvas = document.createElement('canvas');
        chartContainer.innerHTML = '';
        chartContainer.appendChild(canvas);
        
        const config = getChartConfig(type, columns, data);
        
        anime({
            targets: chartContainer,
            opacity: [0, 1],
            scale: [0.9, 1],
            duration: 600,
            easing: 'easeOutExpo',
            complete: () => {
                chartContainer.classList.remove('loading');
                chartContainer.classList.add('loaded');
                currentChart = new Chart(
                    canvas.getContext('2d'),
                    config
                );
                
                anime({
                    targets: canvas,
                    opacity: [0, 1],
                    duration: 800,
                    easing: 'easeOutExpo'
                });
            }
        });
    }

    function getChartConfig(type, columns, data) {
        const labelColumn = columns.find(col => 
            !data.some(item => typeof item[col] === 'number')) || columns[0];
        
        const valueColumns = columns.filter(col => 
            col !== labelColumn && 
            data.some(item => typeof item[col] === 'number'));
        
        const isMultiDataset = type === 'line' || valueColumns.length > 1;
        const colors = getChartColors(valueColumns.length);
        
        const labels = data.map(item => {
            const value = item[labelColumn];
            if (value instanceof Date) {
                return value.toLocaleDateString();
            }
            return value;
        });
        
        const config = {
            type: type,
            data: {
                labels: labels,
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: isMultiDataset ? 'index' : 'point',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== undefined) {
                                    label += context.parsed.y.toLocaleString();
                                } else {
                                    label += context.raw.toLocaleString();
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        }
                    }
                }
            }
        };
        
        if (isMultiDataset) {
            valueColumns.forEach((col, i) => {
                config.data.datasets.push({
                    label: col,
                    data: data.map(item => item[col]),
                    backgroundColor: type === 'pie' ? 
                        colors[i].replace(')', ', 0.7)').replace('rgb', 'rgba') : 
                        colors[i],
                    borderColor: colors[i],
                    borderWidth: 1,
                    tension: 0.1
                });
            });
        } else {
            const valueCol = valueColumns[0] || columns.find(col => col !== labelColumn);
            config.data.datasets.push({
                label: valueCol || 'Value',
                data: data.map(item => item[valueCol]),
                backgroundColor: type === 'pie' ? 
                    data.map((_, i) => colors[i % colors.length]) : 
                    colors[0],
                borderColor: colors[0],
                borderWidth: 1,
                tension: 0.1
            });
        }
        
        return config;
    }

    function getChartColors(count) {
        const palette = [
            'rgb(78, 115, 223)',
            'rgb(28, 200, 138)',
            'rgb(54, 185, 204)',
            'rgb(246, 194, 62)',
            'rgb(231, 74, 59)',
            'rgb(133, 135, 150)',
            'rgb(90, 92, 105)',
            'rgb(0, 204, 153)',
            'rgb(153, 102, 255)',
            'rgb(255, 153, 51)'
        ];
        return palette.slice(0, count);
    }

    function toggleDataView(columns, data) {
        const dataTable = document.getElementById('dataTable');
        if (dataTable) {
            dataTable.classList.toggle('hidden-content');
            
            anime({
                targets: dataTable,
                opacity: dataTable.classList.contains('hidden-content') ? [1, 0] : [0, 1],
                height: dataTable.classList.contains('hidden-content') ? [dataTable.scrollHeight, 0] : [0, dataTable.scrollHeight],
                duration: 300,
                easing: 'easeInOutSine'
            });
        } else {
            const newTable = createDataTable(columns, data);
            newTable.className += ' mt-3';
            newTable.style.opacity = 0;
            newTable.style.height = '0';
            document.querySelector('.bot-message:last-child').appendChild(newTable);
            
            anime({
                targets: newTable,
                opacity: [0, 1],
                height: [0, newTable.scrollHeight],
                duration: 300,
                easing: 'easeInOutSine'
            });
        }
    }

    function createDataTable(columns, data) {
        const tableDiv = document.createElement('div');
        tableDiv.className = 'table-responsive';
        
        const table = document.createElement('table');
        table.className = 'table table-sm table-hover';
        
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        const tbody = document.createElement('tbody');
        data.forEach(row => {
            const tr = document.createElement('tr');
            columns.forEach(col => {
                const td = document.createElement('td');
                const cellValue = row[col];
                td.textContent = cellValue !== null ? cellValue : 'NULL';
                if (typeof cellValue === 'number') {
                    td.className = 'text-end';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        tableDiv.appendChild(table);
        
        return tableDiv;
    }

    function toggleQueryView(query) {
        const queryDiv = document.getElementById('queryContent');
        if (queryDiv) {
            queryDiv.classList.toggle('hidden-content');
            
            anime({
                targets: queryDiv,
                opacity: queryDiv.classList.contains('hidden-content') ? [1, 0] : [0, 1],
                height: queryDiv.classList.contains('hidden-content') ? [queryDiv.scrollHeight, 0] : [0, queryDiv.scrollHeight],
                duration: 300,
                easing: 'easeInOutSine'
            });
        } else {
            const newQueryDiv = document.createElement('pre');
            newQueryDiv.className = 'hidden-content bg-light p-3 rounded mt-3';
            newQueryDiv.id = 'queryContent';
            newQueryDiv.textContent = query;
            newQueryDiv.style.opacity = 0;
            newQueryDiv.style.height = '0';
            document.querySelector('.bot-message:last-child').appendChild(newQueryDiv);
            
            anime({
                targets: newQueryDiv,
                opacity: [0, 1],
                height: [0, newQueryDiv.scrollHeight],
                duration: 300,
                easing: 'easeInOutSine'
            });
        }
    }

    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            const notification = document.createElement('div');
            notification.className = 'notification';
            notification.innerHTML = `
                <i class="bi bi-check-circle-fill"></i>
                <span>Query copied to clipboard!</span>
            `;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.classList.add('hide');
                setTimeout(() => {
                    notification.remove();
                }, 300);
            }, 2000);
        });
    }

    async function downloadResults(queryId) {
        try {
            const downloadBtn = document.querySelector('.btn-download');
            const originalHtml = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '<i class="bi bi-hourglass"></i> Preparing...';
            downloadBtn.disabled = true;
            
            const response = await fetch(`/api/query/${queryId}/download`);
            if (!response.ok) throw new Error('Download failed');
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `query_results_${queryId.slice(0, 8)}.csv`;
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            downloadBtn.innerHTML = '<i class="bi bi-check-circle"></i> Downloaded!';
            anime({
                targets: downloadBtn,
                backgroundColor: ['#ecfdf5', '#059669'],
                duration: 800,
                easing: 'easeOutExpo',
                complete: () => {
                    setTimeout(() => {
                        downloadBtn.innerHTML = originalHtml;
                        downloadBtn.disabled = false;
                        anime({
                            targets: downloadBtn,
                            backgroundColor: ['#059669', '#ecfdf5'],
                            duration: 500,
                            easing: 'easeOutExpo'
                        });
                    }, 1500);
                }
            });
        } catch (error) {
            console.error('Download failed:', error);
            
            const downloadBtn = document.querySelector('.btn-download');
            const originalHtml = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '<i class="bi bi-x-circle"></i> Failed';
            downloadBtn.disabled = true;
            
            anime({
                targets: downloadBtn,
                backgroundColor: ['#fee2e2', '#dc2626'],
                duration: 500,
                easing: 'easeOutExpo',
                complete: () => {
                    setTimeout(() => {
                        downloadBtn.innerHTML = originalHtml;
                        downloadBtn.disabled = false;
                        anime({
                            targets: downloadBtn,
                            backgroundColor: ['#dc2626', '#ecfdf5'],
                            duration: 500,
                            easing: 'easeOutExpo'
                        });
                    }, 1500);
                }
            });
        }
    }

    function updatePollingStatus(result, attempt) {
        if (result.process_log?.length > 0) {
            const lastLog = result.process_log[result.process_log.length - 1];
            loadingStatus.textContent = lastLog;
            
            anime({
                targets: loadingStatus,
                scale: [1.05, 1],
                duration: 300,
                easing: 'easeOutSine'
            });
        } else {
            loadingStatus.textContent = `Processing (attempt ${attempt})...`;
        }
    }

    function handleFailedQuery() {
        removeTypingIndicator();
        resetSubmitButton();
        addMessage("Sorry, I couldn't process your query. Please try a different question or rephrase your request.", 'bot');
        loadingModal.hide();
    }

    function handlePollingTimeout() {
        removeTypingIndicator();
        resetSubmitButton();
        addMessage("The query is taking longer than expected. Please try again later or ask a simpler question.", 'bot');
        loadingModal.hide();
    }

    function handlePollingError(error) {
        console.error('Polling error:', error);
        removeTypingIndicator();
        resetSubmitButton();
        
        if (error.message === 'Result not ready') {
            addMessage("Results are taking longer than expected. Please try again later.", 'bot');
        } else {
            addMessage("Error fetching results. Please refresh and try again.", 'bot');
        }
        
        loadingModal.hide();
    }

    function handleQueryError(error) {
        console.error('Query submission failed:', error);
        removeTypingIndicator();
        resetSubmitButton();
        addMessage("Failed to submit query. Please check your connection and try again.", 'bot');
        loadingModal.hide();
    }

    function startNewChat() {
        anime({
            targets: chatContainer.children,
            opacity: [1, 0],
            translateX: [0, 50],
            duration: 300,
            easing: 'easeInExpo',
            delay: anime.stagger(100, {start: 100}),
            complete: () => {
                chatContainer.innerHTML = '';
                emptyState.style.display = 'flex';
                
                anime({
                    targets: emptyState,
                    opacity: [0, 1],
                    scale: [0.9, 1],
                    duration: 500,
                    easing: 'easeOutExpo'
                });
            }
        });
        
        currentQueryId = null;
        
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }
        
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    window.insertExample = function(query) {
        questionInput.value = query;
        questionInput.focus();
    };
});