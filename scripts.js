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
        });
    });
}

// Função para implementar zoom nas imagens
function setupImageZoom() {
    const images = document.querySelectorAll('.model-viz img, .results-viz img, .optimization-viz img');
    
    images.forEach(img => {
        img.addEventListener('click', () => {
            img.classList.toggle('zoomed');
            
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
                document.body.appendChild(overlay);
                
                // Fechar o zoom ao clicar no overlay
                overlay.addEventListener('click', () => {
                    img.classList.remove('zoomed');
                    overlay.remove();
                });
            } else {
                // Remover overlay quando o zoom for desativado
                const overlay = document.querySelector('.zoom-overlay');
                if (overlay) overlay.remove();
            }
        });
    });
}

// Inicializar as funções quando o DOM estiver carregado
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupImageZoom();
    
    // Detectar quando novas imagens são carregadas para aplicar zoom
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                setupImageZoom();
            }
        });
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
});

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