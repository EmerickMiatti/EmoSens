const emotionIcons = {
    'admiration': '👏', 'amusement': '😄', 'anger': '😡', 'annoyance': '😒',
    'approval': '👍', 'caring': '🤗', 'confusion': '😕', 'curiosity': '🤔',
    'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
    'embarrassment': '😳', 'excitement': '🤩', 'fear': '😨', 'gratitude': '🙏',
    'grief': '😢', 'joy': '😊', 'love': '❤️', 'nervousness': '😰',
    'optimism': '🌟', 'pride': '😤', 'realization': '💡', 'relief': '😌',
    'remorse': '😔', 'sadness': '😭', 'surprise': '😮', 'neutral': '😐'
};

const textInput = document.getElementById('textInput');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');
const emotionsList = document.getElementById('emotionsList');

// Mise à jour de l'affichage du seuil
thresholdSlider.addEventListener('input', (e) => {
    const value = (parseFloat(e.target.value) * 100).toFixed(0);
    thresholdValue.textContent = value + '%';
});

// Boutons d'exemples
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        textInput.value = btn.dataset.text;
        textInput.focus();
    });
});

// Bouton effacer
clearBtn.addEventListener('click', () => {
    textInput.value = '';
    results.classList.remove('show');
    error.classList.remove('show');
});

// Bouton analyser
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();

    if (!text) {
        showError('Veuillez entrer un texte à analyser');
        return;
    }

    // Afficher le chargement
    loading.classList.add('show');
    results.classList.remove('show');
    error.classList.remove('show');
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                threshold: parseFloat(thresholdSlider.value)
            })
        });

        if (!response.ok) {
            throw new Error('Erreur API: ' + response.status);
        }

        const data = await response.json();
        displayResults(data);

    } catch (err) {
        showError('Erreur: ' + err.message + '. Assurez-vous que l\'API est lancée (python app.py)');
    } finally {
        loading.classList.remove('show');
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    const emotions = data.emotions;

    if (emotions.length === 0) {
        emotionsList.innerHTML = '<div class="no-emotions">Aucune émotion détectée au seuil actuel. Essayez de baisser le seuil.</div>';
        document.getElementById('emotionsCount').textContent = '0';
        document.getElementById('maxScore').textContent = '0%';
        document.getElementById('avgScore').textContent = '0%';
    } else {
        const scores = emotions.map(e => e.probability * 100);
        const maxScore = Math.max(...scores);
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

        emotionsList.innerHTML = emotions.map(emotion => `
            <div class="emotion-item">
                <span class="emotion-icon">${emotionIcons[emotion.emotion] || '🎭'}</span>
                <span class="emotion-name">${emotion.emotion}</span>
                <div class="emotion-bar-container">
                    <div class="emotion-bar" style="width: ${emotion.probability * 100}%"></div>
                </div>
                <span class="emotion-value">${(emotion.probability * 100).toFixed(1)}%</span>
            </div>
        `).join('');

        document.getElementById('emotionsCount').textContent = emotions.length;
        document.getElementById('maxScore').textContent = maxScore.toFixed(1) + '%';
        document.getElementById('avgScore').textContent = avgScore.toFixed(1) + '%';
    }

    results.classList.add('show');
}

function showError(message) {
    error.textContent = message;
    error.classList.add('show');
}

textInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeBtn.click();
    }
});
