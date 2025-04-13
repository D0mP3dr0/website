// Função para alternar entre abas nas interfaces com abas
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    if (tabButtons.length === 0) return;

    // Ativar a primeira aba por padrão
    tabButtons[0].classList.add('active');
    tabContents[0].classList.add('active');

    // Adicionar listeners para todos os botões de aba
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remover estado ativo de todos os botões e conteúdos
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Ativar o botão atual e o conteúdo correspondente
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Efeito sonoro cyberpunk ao mudar de aba
            playUISound('tab');
        });
    });
}

// Função para implementar zoom nas imagens
function setupImageZoom() {
    const images = document.querySelectorAll('.model-viz img, .results-viz img, .optimization-viz img');
    
    images.forEach(img => {
        img.addEventListener('click', () => {
            img.classList.toggle('zoomed');
            
            // Som de interface ao ampliar
            playUISound('zoom');
            
            // Adicionar um overlay quando a imagem estiver ampliada
            if (img.classList.contains('zoomed')) {
                const overlay = document.createElement('div');
                overlay.classList.add('zoom-overlay');
                overlay.style.position = 'fixed';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0,0,0,0.7)';
                overlay.style.zIndex = '999';
                
                // Adicionar efeito de scan na imagem ampliada
                const scanEffect = document.createElement('div');
                scanEffect.classList.add('scan-effect');
                overlay.appendChild(scanEffect);
                
                document.body.appendChild(overlay);
                
                // Fechar o zoom ao clicar no overlay
                overlay.addEventListener('click', () => {
                    img.classList.remove('zoomed');
                    overlay.remove();
                    playUISound('zoom-out');
                });
            } else {
                // Remover overlay quando o zoom for desativado
                const overlay = document.querySelector('.zoom-overlay');
                if (overlay) overlay.remove();
            }
        });
    });
}

// Função para inicializar as abas nos mapas interativos
function initMapTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Encontrar o ID da aba alvo
            const tabId = button.getAttribute('data-tab');
            
            // Remover a classe 'active' de todos os botões e conteúdos
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Adicionar a classe 'active' ao botão clicado
            button.classList.add('active');
            
            // Ativar o conteúdo correspondente
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
}

// Função para inicializar as abas de análises detalhadas
function initDetailedTabs() {
    const detailTabButtons = document.querySelectorAll('.detailed-tabs .tab-btn');
    
    detailTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Encontrar o ID da aba alvo
            const tabId = button.getAttribute('data-tab');
            
            // Remover a classe 'active' de todos os botões e conteúdos
            document.querySelectorAll('.detailed-tabs .tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            document.querySelectorAll('.detailed-tabs .tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Adicionar a classe 'active' ao botão clicado
            button.classList.add('active');
            
            // Ativar o conteúdo correspondente
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Função para criar efeito de chuva de código binário
function createBinaryRain() {
    const binaryRain = document.createElement('div');
    binaryRain.className = 'binary-rain';
    document.body.appendChild(binaryRain);
    
    function createBinaryDrop() {
        if (document.hidden) return;
        
        const drop = document.createElement('div');
        drop.className = 'binary-drop';
        // Variação com código binário e hexadecimal para aspecto mais cyberpunk
        const symbols = ['0', '1', 'A', 'F', '7', 'E', '3', 'D'];
        drop.textContent = symbols[Math.floor(Math.random() * symbols.length)];
        drop.style.left = `${Math.random() * 100}%`;
        drop.style.animationDuration = `${5 + Math.random() * 15}s`;
        drop.style.opacity = `${0.1 + Math.random() * 0.4}`;
        drop.style.color = Math.random() > 0.8 ? '#00f0ff' : '#ffffff';
        drop.style.textShadow = Math.random() > 0.9 ? '0 0 5px #ff00f7' : 'none';
        
        binaryRain.appendChild(drop);
        
        // Remover a gota após a animação
        setTimeout(() => {
            drop.remove();
        }, 20000);
    }
    
    // Criar gotas iniciais
    for (let i = 0; i < 40; i++) {
        setTimeout(createBinaryDrop, i * 100);
    }
    
    // Continuar criando gotas em intervalos
    setInterval(createBinaryDrop, 300);
}

// Função para criar efeito de scanline
function createScanlineEffect() {
    const scanline = document.createElement('div');
    scanline.className = 'scanline';
    document.body.appendChild(scanline);
}

// Função para criar efeito de grid digital
function createDigitalGrid() {
    const dataGrid = document.createElement('div');
    dataGrid.className = 'data-grid';
    document.body.appendChild(dataGrid);
}

// Função para adicionar efeitos de hover nos headings
function setupHeadingEffects() {
    const headings = document.querySelectorAll('h1, h2, h3');
    
    headings.forEach(heading => {
        heading.addEventListener('mouseenter', () => {
            // Adicionando efeito de glitch ao passar o mouse
            heading.style.animation = 'glitch 0.3s cubic-bezier(.25, .46, .45, .94) both';
            heading.style.animationIterationCount = '1';
            
            // Som de interface cyberpunk
            playUISound('glitch');
            
            setTimeout(() => {
                heading.style.animation = '';
            }, 300);
        });
    });
}

// Sistema de sons de interface cyberpunk
function initAudioSystem() {
    // Criar elementos de áudio
    const sounds = {
        'hover': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADwAD///////////////////////////////////////////8AAAA8TEFNRTMuMTAwBK8AAAAAAAAAABUgJAJAQQABmgCdwECAT/////////+hNf///////wAAADttdGF0AAAADHZlcmIAAAAAAAAABm10ZGF0AAAAHm1kYXQAAAAAAe0D8wP/2f/Z/e09UPMQww5qgMpjI4cZuHMqaO1IqaORDCeMzCZSiTPbCb8fT6xRy7/eWGjn+8eXS3/VdM/M5pEQD4SHhoWOlVRqHAMAAC8AAB8AAAgAABEAABAAAG8AAAPzA/PYqVYaUe19ZGNTu9Xl3+V/6vv//q5eULDy+gAALwAAHwAACAAAEQAAEAAAbwAAA/MD89ipVhpR7dDy/n//////83//8sACADAAAAIYhBhIJGBAcAAAAAAAAAAAAAAAAAAAAAAAAAAACQAA',
        'click': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADwAD///////////////////////////////////////////8AAAA8TEFNRTMuMTAwBK8AAAAAAAAAABUgJAJAQQABmgCdwECAT/////////+hNf///////wAAADttdGF0AAAADHZlcmIAAAAAAAAABm10ZGF0AAAAHm1kYXQAAAAAAe0D8wP/2f/Z/e09UhDDJSHjjSHDi2AXjlNwHGe4iJiTMuB8eJdTBfWbgXN4KpfP95Z50N/r1F7lnqQ/4eqmn93V//vLlWH39QwHCQ0TN1csTh8jWR4AAC8AAB8AAAgAABEAABAAAG8AAAPzA/PZfVmZW7vPLDC1tPtTvLsrn/+7r/XdVysFHF9AAAXgAALwAAHwAACAAAEQAAEAAAbwAAA/MD89l9WZlbu88sIrWb/////////zVVVVVcrA',
        'tab': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADwAD///////////////////////////////////////////8AAAA8TEFNRTMuMTAwBK8AAAAAAAAAABUgJAJAQQABmgCdwECAT/////////+hNf///////wAAADttdGF0AAAADHZlcmIAAAAAAAAABm10ZGF0AAAAHm1kYXQAAAAAAe0D8wP/2f/Z/e09VUhEzQppcTjcDo0Jh+OdGMajMJ2BwT80YTCbSjg0pF7i6nMT/eWeXH+ssy5M//VTMz/qqZeX/Vb+H+rM0/6YYcOHDX/PfnwoAABvAAAfAAAIAAARAAAQAABvAAAD8wPz2azSytnd3ljFjjsrluX5nNdrv9X9jGjLRVgKAAAcXEoAABvAAAfAAAIAAARAAAQAABvAAAD8wPz2azSytnd3liKNzTf///9JImZarUSAAAAAAAAAAA',
        'glitch': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADwAD///////////////////////////////////////////8AAAA8TEFNRTMuMTAwBK8AAAAAAAAAABUgJAJAQQABmgCdwECAT/////////+hNf///////wAAADttdGF0AAAADHZlcmIAAAAAAAAABm10ZGF0AAAAHm1kYXQAAAAAAe0D8wP/2f/Z/e09WofDCcWH444TJ4AQtE3AcOLeCnDMtw4nhPGU4+Pu8F2I4/3lnkbX+rfPrl5f9Vav+rmeqz+GHnqsxpTDh81mff38QAAAgAAB8AAAgAABEAABAAAG8AAAPzA/PZtdy46NdT3MVxYzXW93f6u7u/96zdbKx4sAACAAAHwAACAAAEQAAEAAAbwAAA/MD89m13Llo11PcxXFqS7//////2qJJJG2VioAAAAAAAAAA',
        'zoom': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADwAD///////////////////////////////////////////8AAAA8TEFNRTMuMTAwBK8AAAAAAAAAABUgJAJAQQABmgCdwECAT/////////+hNf///////wAAADttdGF0AAAADHZlcmIAAAAAAAAABm10ZGF0AAAAHm1kYXQAAAAAAe0D8wP/2f/Z/e09YcVIcOHIAOHnzjM3A8GLuOHDzvCZj8ZmPjzvD58L5tNwLpWH+8s8mb/rNMzn/jOXv/yQ/hhuXHbV2G5l/vBAAQAAAHwAACAAAEQAAEAAAbwAAA/MD89o12muVpfaxjDZK1Vdfv9f/+KuzQVTyQAEAAAAfAAAIAAARAAAgAADbwAAD8wPz2jXaa5Wl9rGMjqS///1f//83MzAqnk',
        'zoom-out': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADwAD///////////////////////////////////////////8AAAA8TEFNRTMuMTAwBK8AAAAAAAAAABUgJAJAQQABmgCdwECAT/////////+hNf///////wAAADttdGF0AAAADHZlcmIAAAAAAAAABm10ZGF0AAAAHm1kYXQAAAAAAe0D8wP/2f/Z/e09YcVIcOHIAOHnzjM3A8GLuOHDzvCZj8ZmPjzvD58L5tNwLpWH+8s8mb/rNMzn/jOXv/yQ/hhuXHbV2G5l/vBAAQAAAHwAACAAAEQAAEAAAbwAAA/MD89o12muVpfaxjDZK1Vdfv9f/+KuzQVTyQAEAAAAfAAAIAAARAAAgAADbwAAD8wPz2jXaa5Wl9rGMjqS///1f//83MzAqnk'
    };
    
    const audioElements = {};
    
    // Criar elementos de áudio para cada som
    for (const [key, src] of Object.entries(sounds)) {
        const audio = new Audio(src);
        audio.volume = 0.2; // Volume baixo para não incomodar
        audioElements[key] = audio;
    }
    
    // Armazenar globalmente
    window.uiSounds = audioElements;
}

// Função para reproduzir sons de interface
function playUISound(type) {
    if (!window.uiSounds) return;
    
    const sound = window.uiSounds[type];
    if (sound) {
        // Reiniciar o som para permitir reproduções rápidas consecutivas
        sound.currentTime = 0;
        sound.play().catch(err => console.log('Erro ao tocar som: ', err));
    }
}

// Função para criar efeito de digitação em tempo real
function typewriterEffect(element, text, speed = 50) {
    let i = 0;
    element.textContent = '';
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            
            // Som de digitação aleatório
            if (Math.random() > 0.7) {
                playUISound('hover');
            }
            
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Função para criar efeitos de falha (glitch) aleatórios nos elementos
function setupRandomGlitches() {
    const elements = document.querySelectorAll('.feature-card, .result-card, .tool-card, .analysis-card');
    
    function applyRandomGlitch() {
        if (document.hidden) return;
        
        const randomElement = elements[Math.floor(Math.random() * elements.length)];
        
        // Não aplicar se já estiver com glitch
        if (randomElement.classList.contains('glitching')) return;
        
        randomElement.classList.add('glitching');
        
        // Aplicar efeito visual
        const originalTransform = randomElement.style.transform;
        const originalBoxShadow = randomElement.style.boxShadow;
        
        randomElement.style.transform = `${originalTransform} translate(${(Math.random() * 4) - 2}px, ${(Math.random() * 4) - 2}px)`;
        randomElement.style.boxShadow = `0 0 10px rgba(0, 240, 255, 0.7), 0 0 20px rgba(255, 0, 247, 0.4)`;
        
        // Restaurar após um curto período
        setTimeout(() => {
            randomElement.style.transform = originalTransform;
            randomElement.style.boxShadow = originalBoxShadow;
            randomElement.classList.remove('glitching');
        }, 150 + Math.random() * 200);
    }
    
    // Aplicar glitches aleatórios em intervalos
    setInterval(applyRandomGlitch, 3000);
}

// Função para adicionar efeito de partículas nos botões
function setupButtonParticles() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            // Criar partículas
            for (let i = 0; i < 10; i++) {
                createParticle(button);
            }
            playUISound('hover');
        });
        
        button.addEventListener('click', () => {
            playUISound('click');
        });
    });
    
    function createParticle(element) {
        const particle = document.createElement('div');
        const rect = element.getBoundingClientRect();
        
        // Posição aleatória ao redor do botão
        const x = rect.left + rect.width * Math.random();
        const y = rect.top + rect.height * Math.random();
        
        // Estilo da partícula
        particle.style.position = 'fixed';
        particle.style.left = `${x}px`;
        particle.style.top = `${y}px`;
        particle.style.width = '2px';
        particle.style.height = '2px';
        particle.style.backgroundColor = 'rgba(0, 240, 255, 0.7)';
        particle.style.borderRadius = '50%';
        particle.style.pointerEvents = 'none';
        particle.style.zIndex = '9999';
        
        // Animação
        particle.animate([
            { 
                transform: `translate(0, 0)`,
                opacity: 1 
            },
            { 
                transform: `translate(${(Math.random() * 100) - 50}px, ${(Math.random() * 100) - 50}px)`,
                opacity: 0 
            }
        ], {
            duration: 1000,
            easing: 'cubic-bezier(0.1, 0.8, 0.2, 1)'
        });
        
        document.body.appendChild(particle);
        
        // Remover após a animação
        setTimeout(() => {
            particle.remove();
        }, 1000);
    }
}

// Adicionar efeito de cursor cyberpunk
function setupCyberpunkCursor() {
    const cursor = document.createElement('div');
    cursor.classList.add('cyberpunk-cursor');
    document.body.appendChild(cursor);
    
    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    });
    
    document.addEventListener('mousedown', () => {
        cursor.classList.add('active');
    });
    
    document.addEventListener('mouseup', () => {
        cursor.classList.remove('active');
    });
    
    // Adiciona scanline com z-index menor
    const scanline = document.createElement('div');
    scanline.classList.add('scanline');
    scanline.style.zIndex = '1';
    document.body.appendChild(scanline);
    
    // Esconde o cursor padrão para evitar duplicação
    document.body.style.cursor = 'none';
    
    // Corrige os links e botões para mostrar cursor de ponteiro personalizado
    const clickableElements = document.querySelectorAll('a, button, .btn, .card, .feature-card, .result-card, .tool-card');
    clickableElements.forEach(el => {
        el.style.cursor = 'none';
        el.addEventListener('mouseenter', () => {
            cursor.classList.add('hovering');
        });
        el.addEventListener('mouseleave', () => {
            cursor.classList.remove('hovering');
        });
    });
}

// Função para criar efeito de análise de IA
function createAIAnalysisEffect() {
    const neuralNetworkBg = document.querySelector('.neural-network-bg');
    if (!neuralNetworkBg) return;
    
    // Criar nós de rede neural
    for (let i = 0; i < 50; i++) {
        const node = document.createElement('div');
        node.className = 'neural-node';
        node.style.left = `${Math.random() * 100}%`;
        node.style.top = `${Math.random() * 100}%`;
        node.style.width = `${4 + Math.random() * 8}px`;
        node.style.height = node.style.width;
        node.style.opacity = 0.1 + Math.random() * 0.5;
        
        neuralNetworkBg.appendChild(node);
    }
    
    // Criar conexões entre os nós
    const nodes = neuralNetworkBg.querySelectorAll('.neural-node');
    
    for (let i = 0; i < 80; i++) {
        const connection = document.createElement('div');
        connection.className = 'neural-connection';
        
        // Pegar dois nós aleatórios
        const nodeA = nodes[Math.floor(Math.random() * nodes.length)];
        const nodeB = nodes[Math.floor(Math.random() * nodes.length)];
        
        // Calcular a posição e ângulo da conexão
        const rectA = nodeA.getBoundingClientRect();
        const rectB = nodeB.getBoundingClientRect();
        
        const neuralRect = neuralNetworkBg.getBoundingClientRect();
        
        const startX = (rectA.left - neuralRect.left) + (rectA.width / 2);
        const startY = (rectA.top - neuralRect.top) + (rectA.height / 2);
        const endX = (rectB.left - neuralRect.left) + (rectB.width / 2);
        const endY = (rectB.top - neuralRect.top) + (rectB.height / 2);
        
        const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
        const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
        
        connection.style.width = `${length}px`;
        connection.style.left = `${startX}px`;
        connection.style.top = `${startY}px`;
        connection.style.transform = `rotate(${angle}deg)`;
        connection.style.opacity = 0.1 + Math.random() * 0.3;
        
        neuralNetworkBg.appendChild(connection);
        
        // Animar a conexão
        setInterval(() => {
            const pulseEffect = document.createElement('div');
            pulseEffect.className = 'data-pulse';
            pulseEffect.style.left = '0';
            pulseEffect.style.opacity = '0.8';
            
            connection.appendChild(pulseEffect);
            
            setTimeout(() => {
                pulseEffect.style.left = '100%';
                pulseEffect.style.opacity = '0';
            }, 50);
            
            setTimeout(() => {
                pulseEffect.remove();
            }, 1000);
        }, 2000 + Math.random() * 5000);
    }
}

// Função para animar os indicadores de processamento
function animateProcessors() {
    const processors = document.querySelectorAll('.data-processor');
    
    processors.forEach(processor => {
        const lights = processor.querySelectorAll('.light');
        const label = processor.querySelector('.processor-label');
        
        // Animar as luzes
        let activeLight = 0;
        setInterval(() => {
            lights.forEach(light => light.classList.remove('active'));
            lights[activeLight].classList.add('active');
            activeLight = (activeLight + 1) % lights.length;
        }, 500);
        
        // Animar o texto com efeito de digitação
        const states = [
            "System: Online",
            "Processing: Active",
            "Neural Net: Training",
            "Data Analysis: Running"
        ];
        
        let stateIndex = 0;
        setInterval(() => {
            if (label) {
                const nextState = states[stateIndex];
                typewriterEffect(label, nextState, 20);
                stateIndex = (stateIndex + 1) % states.length;
            }
        }, 5000);
    });
}

// Adicionar animação aos pontos de carregamento
function animateLoadingDots() {
    const loadingContainers = document.querySelectorAll('.loading-ai');
    
    loadingContainers.forEach(container => {
        const dots = container.querySelectorAll('.dot');
        let activeDot = 0;
        
        setInterval(() => {
            dots.forEach(dot => dot.classList.remove('active'));
            dots[activeDot].classList.add('active');
            activeDot = (activeDot + 1) % dots.length;
        }, 300);
    });
}

// Função para adicionar efeito de dados nas métricas
function animateMetrics() {
    const metrics = document.querySelectorAll('.metric-value');
    
    metrics.forEach(metric => {
        const originalText = metric.textContent;
        let currentValue = parseFloat(originalText.replace(/[^\d.]/g, ''));
        const suffix = originalText.replace(/[\d.]/g, '');
        
        // Animar ocasionalmente com pequenas flutuações
        setInterval(() => {
            if (Math.random() > 0.7) {
                // Pequena variação aleatória
                const variation = currentValue * 0.03 * (Math.random() > 0.5 ? 1 : -1);
                const newValue = Math.max(0, currentValue + variation);
                
                // Efeito de digitação ao atualizar
                metric.classList.add('updating');
                setTimeout(() => {
                    metric.textContent = newValue.toFixed(1).replace(/\.0$/, '') + suffix;
                    metric.classList.remove('updating');
                }, 200);
                
                // Resetar para o valor original após algum tempo
                setTimeout(() => {
                    metric.textContent = originalText;
                    currentValue = parseFloat(originalText.replace(/[^\d.]/g, ''));
                }, 2000);
            }
        }, 3000);
    });
}

// Função para trocar os estilos de tema dinamicamente
function setupThemeSwitching() {
    const themeToggle = document.createElement('button');
    themeToggle.classList.add('theme-toggle');
    themeToggle.innerHTML = '<i class="fas fa-adjust"></i>';
    document.body.appendChild(themeToggle);
    
    themeToggle.addEventListener('click', () => {
        // Adiciona o efeito de flash
        const flash = document.createElement('div');
        flash.classList.add('theme-flash');
        document.body.appendChild(flash);
        
        setTimeout(() => {
            flash.remove();
        }, 500);
        
        // Alterna a classe de tema
        document.body.classList.toggle('light-theme');
        
        // Salva a preferência
        const isLightTheme = document.body.classList.contains('light-theme');
        localStorage.setItem('lightTheme', isLightTheme);
    });
    
    // Verifica a preferência salva
    const savedTheme = localStorage.getItem('lightTheme');
    if (savedTheme === 'true') {
        document.body.classList.add('light-theme');
    }
}

// Inicializar as funções quando o DOM estiver carregado
document.addEventListener('DOMContentLoaded', function() {
    // Inicializar sistema de áudio
    initAudioSystem();
    
    setupTabs();
    setupImageZoom();
    initMapTabs();
    initDetailedTabs();
    setupHeadingEffects();
    setupButtonParticles();
    setupCyberpunkCursor();
    
    // Efeitos cyberpunk
    createBinaryRain();
    createScanlineEffect();
    createDigitalGrid();
    setupRandomGlitches();
    
    // Novas funções adicionadas
    createAIAnalysisEffect();
    animateProcessors();
    animateLoadingDots();
    animateMetrics();
    setupThemeSwitching();
    
    // Efetio de digitação na frase principal do hero
    const heroSubtitle = document.querySelector('.subtitle');
    if (heroSubtitle) {
        const originalText = heroSubtitle.textContent;
        setTimeout(() => {
            typewriterEffect(heroSubtitle, originalText, 30);
        }, 500);
    }
    
    // Adicionar efeito de "IA processando dados" em elementos de dados
    document.querySelectorAll('.data-section, .stats-container').forEach(section => {
        const dataProcessor = document.createElement('div');
        dataProcessor.className = 'data-processor';
        dataProcessor.innerHTML = `
            <div class="processor-lights">
                <div class="light"></div>
                <div class="light"></div>
                <div class="light"></div>
            </div>
            <div class="processor-label">IA processando dados</div>
        `;
        section.appendChild(dataProcessor);
    });
    
    // Detectar quando novas imagens são carregadas para aplicar zoom
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                setupImageZoom();
            }
        });
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Inicializar efeitos hover nos feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
            this.style.boxShadow = '0 0 15px rgba(0, 240, 255, 0.7), 0 0 30px rgba(0, 240, 255, 0.4)';
            playUISound('hover');
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 6px rgba(0, 240, 255, 0.15), 0 2px 4px rgba(139, 0, 255, 0.1)';
        });
    });
    
    // Adiciona efeitos cyberpunk AI
    createNeuralNetworkBackground();
    createAIProcessorEffects();
    setupAIMetricsAnimation();
    
    // Adiciona efeito Mad Max de poeira
    createMadMaxDustEffect();
    
    // Corrige funcionalidade dos botões
    setupButtonFunctionality();
});

// Função para garantir que os botões funcionem corretamente
function setupButtonFunctionality() {
    // Corrige os botões CTA na seção hero
    const ctaButtons = document.querySelectorAll('.cta-buttons .btn');
    ctaButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href) {
                // Previne comportamento padrão para debugging
                // e.preventDefault();
                
                // Adiciona efeito visual de clique
                this.classList.add('clicked');
                setTimeout(() => {
                    this.classList.remove('clicked');
                }, 300);
                
                // Redireciona apenas se for um link válido (não "#")
                if (href !== '#') {
                    // Log para debugging
                    console.log('Navegando para: ' + href);
                    
                    // Adiciona um pequeno delay para o efeito visual ser notado
                    // setTimeout(() => {
                    //     window.location.href = href;
                    // }, 200);
                }
            }
        });
    });
    
    // Corrige os links de navegação
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href && href !== '#') {
                // Adiciona efeito visual de clique
                this.classList.add('clicked');
                console.log('Navegando para: ' + href);
            }
        });
    });
    
    // Corrige botões nas cards
    const cardButtons = document.querySelectorAll('.feature-card, .result-card, .tool-card');
    cardButtons.forEach(card => {
        card.addEventListener('click', function() {
            // Efeito visual ao clicar
            this.classList.add('pulse');
            setTimeout(() => {
                this.classList.remove('pulse');
            }, 500);
            
            // Se tiver um link dentro do card, ativa-o
            const cardLink = this.querySelector('a');
            if (cardLink) {
                console.log('Navegando para: ' + cardLink.getAttribute('href'));
                // cardLink.click();
            }
        });
    });
    
    // Corrige o botão de tema para garantir que seja clicável
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.style.zIndex = '9999';
        themeToggle.style.cursor = 'pointer';
    }
    
    // Ajusta eventos do cursor para melhorar a interação
    document.addEventListener('mousedown', (e) => {
        // Adiciona highlight de clique na posição exata do clique
        const clickHighlight = document.createElement('div');
        clickHighlight.classList.add('click-highlight');
        clickHighlight.style.left = e.clientX + 'px';
        clickHighlight.style.top = e.clientY + 'px';
        document.body.appendChild(clickHighlight);
        
        // Remove após a animação
        setTimeout(() => {
            clickHighlight.remove();
        }, 600);
    });
}

// Adicionar funcionalidade de rolagem suave aos links internos
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        
        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        
        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop - 80, // Ajuste para considerar o header fixo
                behavior: 'smooth'
            });
        }
    });
});

// Cria o background de rede neural
function createNeuralNetworkBackground() {
    const container = document.createElement('div');
    container.classList.add('neural-network-bg');
    document.body.appendChild(container);
    
    // Número de nós depende do tamanho da janela
    const nodeCount = Math.floor(window.innerWidth * window.innerHeight / 40000);
    const nodes = [];
    
    // Cria os nós
    for (let i = 0; i < nodeCount; i++) {
        const node = document.createElement('div');
        node.classList.add('neural-node');
        
        const size = Math.random() * 4 + 2;
        node.style.width = size + 'px';
        node.style.height = size + 'px';
        
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        node.style.left = x + '%';
        node.style.top = y + '%';
        
        node.style.animationDelay = Math.random() * 5 + 's';
        node.style.opacity = Math.random() * 0.5 + 0.2;
        
        container.appendChild(node);
        nodes.push({node, x, y});
    }
    
    // Cria conexões entre nós próximos
    for (let i = 0; i < nodes.length; i++) {
        const nodeA = nodes[i];
        
        for (let j = i + 1; j < nodes.length; j++) {
            const nodeB = nodes[j];
            const distance = Math.sqrt(Math.pow(nodeB.x - nodeA.x, 2) + Math.pow(nodeB.y - nodeA.y, 2));
            
            // Conecta nós apenas se estiverem a uma distância razoável
            if (distance < 20) {
                const connection = document.createElement('div');
                connection.classList.add('neural-connection');
                
                // Posiciona a conexão
                const length = distance;
                const angle = Math.atan2(nodeB.y - nodeA.y, nodeB.x - nodeA.x) * 180 / Math.PI;
                
                connection.style.width = length + '%';
                connection.style.left = nodeA.x + '%';
                connection.style.top = nodeA.y + '%';
                connection.style.transform = `rotate(${angle}deg)`;
                connection.style.opacity = (1 - distance / 20) * 0.3;
                
                container.appendChild(connection);
                
                // Adiciona pulsos de dados nas conexões aleatoriamente
                if (Math.random() < 0.2) {
                    setInterval(() => {
                        const dataPulse = document.createElement('div');
                        dataPulse.classList.add('data-pulse');
                        
                        dataPulse.style.left = '0%';
                        dataPulse.style.opacity = '0.8';
                        connection.appendChild(dataPulse);
                        
                        setTimeout(() => {
                            dataPulse.style.left = '100%';
                            dataPulse.style.opacity = '0';
                        }, 10);
                        
                        setTimeout(() => {
                            dataPulse.remove();
                        }, 1000);
                    }, Math.random() * 5000 + 3000);
                }
            }
        }
    }
}

// Cria elementos de processador de IA
function createAIProcessorEffects() {
    const sections = document.querySelectorAll('section, .container');
    if (sections.length > 0) {
        // Adiciona 3 processadores na primeira seção
        const targetSection = sections[0];
        const processorContainer = document.createElement('div');
        processorContainer.style.position = 'absolute';
        processorContainer.style.top = '20px';
        processorContainer.style.right = '20px';
        processorContainer.style.zIndex = '10';
        
        // Status de processadores
        const processorStatuses = [
            { name: "Analysis Engine", active: true },
            { name: "Neural Interface", active: false },
            { name: "GeoData Processor", active: true }
        ];
        
        processorStatuses.forEach((status, index) => {
            const processor = document.createElement('div');
            processor.classList.add('data-processor');
            
            // Luzes de status
            const lights = document.createElement('div');
            lights.classList.add('processor-lights');
            
            for (let i = 0; i < 3; i++) {
                const light = document.createElement('div');
                light.classList.add('light');
                if (i === 0 && status.active) {
                    light.classList.add('active');
                }
                lights.appendChild(light);
            }
            
            // Rótulo do processador
            const label = document.createElement('div');
            label.classList.add('processor-label');
            label.textContent = status.name;
            
            // Animação de loading
            const loading = document.createElement('div');
            loading.classList.add('loading-ai');
            
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('div');
                dot.classList.add('dot');
                loading.appendChild(dot);
            }
            
            processor.appendChild(lights);
            processor.appendChild(label);
            processor.appendChild(loading);
            processorContainer.appendChild(processor);
            
            // Anima as luzes
            setInterval(() => {
                const lightsElems = lights.querySelectorAll('.light');
                lightsElems.forEach(light => light.classList.remove('active'));
                
                const randomLight = Math.floor(Math.random() * lightsElems.length);
                lightsElems[randomLight].classList.add('active');
            }, 2000 + index * 500);
            
            // Anima os pontos de loading
            let dotIndex = 0;
            setInterval(() => {
                const dots = loading.querySelectorAll('.dot');
                dots.forEach(dot => dot.classList.remove('active'));
                dots[dotIndex].classList.add('active');
                dotIndex = (dotIndex + 1) % dots.length;
            }, 500);
        });
        
        targetSection.style.position = 'relative';
        targetSection.appendChild(processorContainer);
    }
}

// Animação das métricas de IA
function setupAIMetricsAnimation() {
    // Adiciona uma seção de métricas em páginas adequadas
    if (document.querySelector('.hero')) {
        const hero = document.querySelector('.hero');
        const container = document.createElement('div');
        container.classList.add('ai-metrics');
        
        const metrics = [
            { title: "Accuracy", value: "98.7%", icon: "chart-line", desc: "Neural prediction precision" },
            { title: "Data Points", value: "5.2M", icon: "database", desc: "Geoprocessing samples analyzed" },
            { title: "Response Time", value: "12ms", icon: "bolt", desc: "Average processing latency" },
            { title: "Updates", value: "24/7", icon: "sync", desc: "Continuous model training" }
        ];
        
        metrics.forEach(metric => {
            const card = document.createElement('div');
            card.classList.add('metric-card');
            
            const title = document.createElement('div');
            title.classList.add('metric-title');
            title.innerHTML = `<i class="fas fa-${metric.icon}"></i> ${metric.title}`;
            
            const value = document.createElement('div');
            value.classList.add('metric-value');
            value.textContent = metric.value;
            
            const desc = document.createElement('div');
            desc.classList.add('metric-desc');
            desc.textContent = metric.desc;
            
            card.appendChild(title);
            card.appendChild(value);
            card.appendChild(desc);
            container.appendChild(card);
            
            // Animação de atualização de valores
            setInterval(() => {
                value.classList.add('updating');
                
                setTimeout(() => {
                    value.classList.remove('updating');
                }, 500);
                
                // Pequena flutuação aleatória nos valores
                if (metric.title === "Accuracy") {
                    const base = 98.7;
                    const fluctuation = (Math.random() * 0.4 - 0.2).toFixed(1);
                    value.textContent = `${(base + parseFloat(fluctuation)).toFixed(1)}%`;
                } else if (metric.title === "Response Time") {
                    const base = 12;
                    const fluctuation = Math.floor(Math.random() * 5) - 2;
                    value.textContent = `${Math.max(8, base + fluctuation)}ms`;
                }
            }, 3000 + Math.random() * 2000);
        });
        
        hero.appendChild(container);
    }
}

// Função para criar efeito de poeira no estilo Mad Max
function createMadMaxDustEffect() {
    console.log("Iniciando efeito de poeira Mad Max"); // Debug

    // Remover instância anterior se existir
    const oldContainer = document.querySelector('.mad-max-dust-container');
    if (oldContainer) oldContainer.remove();
    
    const oldGrain = document.querySelector('.grain-overlay');
    if (oldGrain) oldGrain.remove();
    
    const oldAmber = document.querySelector('.amber-overlay');
    if (oldAmber) oldAmber.remove();
    
    const oldStormIndicator = document.querySelector('.storm-indicator');
    if (oldStormIndicator) oldStormIndicator.remove();

    // Cria um indicador visual de que a tempestade está ativa
    const stormIndicator = document.createElement('div');
    stormIndicator.classList.add('storm-indicator');
    stormIndicator.style.position = 'fixed';
    stormIndicator.style.top = '80px';
    stormIndicator.style.right = '20px';
    stormIndicator.style.background = 'rgba(210, 180, 140, 0.3)';
    stormIndicator.style.border = '1px solid rgba(153, 101, 21, 0.8)';
    stormIndicator.style.borderRadius = '4px';
    stormIndicator.style.padding = '5px 10px';
    stormIndicator.style.color = '#fff';
    stormIndicator.style.fontFamily = 'var(--font-heading)';
    stormIndicator.style.fontSize = '12px';
    stormIndicator.style.textTransform = 'uppercase';
    stormIndicator.style.zIndex = '9999';
    stormIndicator.style.boxShadow = '0 0 10px rgba(210, 180, 140, 0.5)';
    stormIndicator.textContent = '🌪️ Modo Mad Max Ativo 🌪️';
    document.body.appendChild(stormIndicator);

    const dustContainer = document.createElement('div');
    dustContainer.classList.add('mad-max-dust-container');
    dustContainer.style.zIndex = '10'; // Garantir que está à frente
    document.body.appendChild(dustContainer);
    
    // Adiciona o overlay de grão de filme
    const grainOverlay = document.createElement('div');
    grainOverlay.classList.add('grain-overlay');
    grainOverlay.style.zIndex = '6'; // Garantir que está visível mas atrás das partículas
    document.body.appendChild(grainOverlay);
    
    // Adiciona um filtro de cor sepia/âmbar para todo o site
    document.documentElement.style.filter = 'sepia(15%)';
    
    // Cria partículas de poeira iniciais
    for (let i = 0; i < 60; i++) { // Aumentei a quantidade
        createDustParticle();
    }
    
    // Continua criando partículas periodicamente
    setInterval(() => {
        if (Math.random() > 0.5) { // Aumentei a probabilidade
            createDustParticle();
        }
    }, 150); // Reduzi o intervalo
    
    function createDustParticle() {
        const dust = document.createElement('div');
        dust.classList.add('dust-particle');
        
        // Posição inicial - sempre na esquerda, mas altura aleatória
        const posY = Math.random() * 100;
        dust.style.top = `${posY}%`;
        dust.style.left = '-20px';
        
        // Tamanho aleatório (aumentei o tamanho)
        const size = Math.random() * 20 + 10;
        dust.style.width = `${size}px`;
        dust.style.height = `${size}px`;
        
        // Opacidade inicial aleatória (aumentei a opacidade)
        dust.style.opacity = Math.random() * 0.4 + 0.2;
        
        // Velocidade aleatória (aumentei a velocidade mínima)
        const speed = Math.random() * 10 + 15;
        
        // Cor aleatória em tons de areia/poeira
        const dustColors = [
            'rgba(210, 180, 140, 0.6)',  // Tan
            'rgba(189, 154, 122, 0.6)',  // Desert Sand
            'rgba(153, 101, 21, 0.5)',   // Amber
            'rgba(179, 139, 109, 0.5)',  // Desert Dust
            'rgba(196, 164, 132, 0.6)'   // Sandstorm
        ];
        dust.style.backgroundColor = dustColors[Math.floor(Math.random() * dustColors.length)];
        
        // Adiciona desfoque para parecer mais realista
        const blurAmount = Math.random() * 3 + 1;
        dust.style.filter = `blur(${blurAmount}px)`;
        
        // Adiciona elemento ao container
        dustContainer.appendChild(dust);
        
        // Define a animação via keyframes
        const animation = dust.animate([
            { left: '-20px', transform: `rotate(0deg) translateY(0px)` },
            { left: `${window.innerWidth + 50}px`, transform: `rotate(${Math.random() * 720}deg) translateY(${(Math.random() - 0.5) * 200}px)` }
        ], {
            duration: speed * 1000,
            easing: 'linear',
            iterations: 1
        });
        
        // Remove partícula após a animação
        animation.onfinish = () => {
            dust.remove();
        };
    }
    
    // Adiciona ondas de poeira mais frequentes
    setInterval(() => {
        if (Math.random() > 0.5) {
            createDustWave();
        }
    }, 5000);
    
    // Cria manchas de areia fixas nas bordas da tela
    createSandPatches();
    
    function createSandPatches() {
        // Adiciona 5 manchas de areia em posições aleatórias
        for (let i = 0; i < 5; i++) {
            const patch = document.createElement('div');
            patch.classList.add('sand-patch');
            
            // Sempre posicionar nas bordas
            const position = Math.floor(Math.random() * 4); // 0: topo, 1: direita, 2: baixo, 3: esquerda
            
            patch.style.position = 'fixed';
            patch.style.zIndex = '7';
            patch.style.width = `${Math.random() * 150 + 50}px`;
            patch.style.height = `${Math.random() * 150 + 50}px`;
            patch.style.borderRadius = '50%';
            patch.style.background = 'radial-gradient(circle, rgba(210, 180, 140, 0.3) 0%, transparent 70%)';
            patch.style.pointerEvents = 'none';
            
            // Posicionar nas bordas
            if (position === 0) { // topo
                patch.style.top = '0';
                patch.style.left = `${Math.random() * 80 + 10}%`;
            } else if (position === 1) { // direita
                patch.style.right = '0';
                patch.style.top = `${Math.random() * 80 + 10}%`;
            } else if (position === 2) { // baixo
                patch.style.bottom = '0';
                patch.style.left = `${Math.random() * 80 + 10}%`;
            } else { // esquerda
                patch.style.left = '0';
                patch.style.top = `${Math.random() * 80 + 10}%`;
            }
            
            document.body.appendChild(patch);
        }
    }
    
    function createDustWave() {
        console.log("Criando onda de poeira"); // Debug
        
        // Cria várias partículas ao mesmo tempo para simular uma onda
        for (let i = 0; i < 25; i++) {
            setTimeout(() => createDustParticle(), i * 40);
        }
        
        // Atualiza o indicador da tempestade para mostrar que uma onda está acontecendo
        stormIndicator.style.backgroundColor = 'rgba(153, 101, 21, 0.6)';
        stormIndicator.style.boxShadow = '0 0 15px rgba(210, 180, 140, 0.7)';
        stormIndicator.textContent = '🌪️ TEMPESTADE DE AREIA 🌪️';
        
        // Restaura após a onda
        setTimeout(() => {
            stormIndicator.style.backgroundColor = 'rgba(210, 180, 140, 0.3)';
            stormIndicator.style.boxShadow = '0 0 10px rgba(210, 180, 140, 0.5)';
            stormIndicator.textContent = '🌪️ Modo Mad Max Ativo 🌪️';
        }, 3000);
        
        // Cria o elemento visual de onda de poeira
        const dustWave = document.createElement('div');
        dustWave.classList.add('dust-wave');
        dustWave.style.zIndex = '8';
        document.body.appendChild(dustWave);
        
        // Remove após a animação
        setTimeout(() => {
            dustWave.remove();
        }, 8000);
    }
    
    // Adiciona uma forte onda de poeira logo no início
    setTimeout(() => {
        createDustWave();
    }, 500);

    // Adiciona outra onda após um curto período
    setTimeout(() => {
        createDustWave();
    }, 3000);
    
    // Aplica o efeito de cor âmbar em todo o site
    const amberOverlay = document.createElement('div');
    amberOverlay.classList.add('amber-overlay');
    amberOverlay.style.position = 'fixed';
    amberOverlay.style.top = '0';
    amberOverlay.style.left = '0';
    amberOverlay.style.width = '100%';
    amberOverlay.style.height = '100%';
    amberOverlay.style.background = 'linear-gradient(to right, rgba(153, 101, 21, 0.08), rgba(210, 180, 140, 0.08))';
    amberOverlay.style.pointerEvents = 'none';
    amberOverlay.style.zIndex = '4';
    document.body.appendChild(amberOverlay);
    
    console.log("Efeito de poeira Mad Max iniciado"); // Debug
} 