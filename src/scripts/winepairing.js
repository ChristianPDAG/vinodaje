// Base de datos de maridajes por tipo de vino
const winePairingDatabase = {
  'malbec': {
    description: 'Vino tinto robusto con notas de frutas oscuras',
    pairings: ['Asado argentino', 'Cordero al horno', 'Quesos maduros', 'Estofado de carne'],
    characteristics: 'Cuerpo completo, taninos suaves, notas de ciruela y cereza negra'
  },
  'cabernet sauvignon': {
    description: 'Vino tinto estructurado con gran potencial de guarda',
    pairings: ['Bife de chorizo', 'Costillas BBQ', 'Queso parmesano', 'Pasta boloñesa'],
    characteristics: 'Cuerpo robusto, taninos firmes, notas de cassis y pimiento verde'
  },
  'cabernet': {
    description: 'Vino tinto estructurado con gran potencial de guarda',
    pairings: ['Bife de chorizo', 'Costillas BBQ', 'Queso parmesano', 'Pasta boloñesa'],
    characteristics: 'Cuerpo robusto, taninos firmes, notas de cassis y pimiento verde'
  },
  'merlot': {
    description: 'Vino tinto suave y aterciopelado',
    pairings: ['Pollo al horno', 'Salmón grillado', 'Risotto de hongos', 'Queso brie'],
    characteristics: 'Cuerpo medio, taninos suaves, notas de ciruela y chocolate'
  },
  'pinot noir': {
    description: 'Vino tinto elegante y delicado',
    pairings: ['Pato laqueado', 'Salmón', 'Champiñones salteados', 'Queso gruyère'],
    characteristics: 'Cuerpo ligero a medio, taninos suaves, notas de cereza y tierra'
  },
  'syrah': {
    description: 'Vino tinto especiado y potente',
    pairings: ['Cordero especiado', 'Carnes a la parrilla', 'Embutidos', 'Queso manchego'],
    characteristics: 'Cuerpo completo, taninos firmes, notas de pimienta negra y mora'
  },
  'chardonnay': {
    description: 'Vino blanco versátil con notas mantecosas',
    pairings: ['Pescado al horno', 'Pollo a la crema', 'Langosta', 'Queso camembert'],
    characteristics: 'Cuerpo medio a completo, notas de manzana verde y vainilla'
  },
  'sauvignon blanc': {
    description: 'Vino blanco fresco y aromático',
    pairings: ['Ensaladas frescas', 'Mariscos', 'Queso de cabra', 'Ceviche'],
    characteristics: 'Cuerpo ligero, acidez vibrante, notas cítricas y herbales'
  },
  'torrontés': {
    description: 'Vino blanco argentino aromático',
    pairings: ['Comida asiática', 'Empanadas', 'Pescado blanco', 'Queso fresco'],
    characteristics: 'Cuerpo ligero, aromático, notas florales y de durazno'
  },
  'tempranillo': {
    description: 'Vino tinto español versátil',
    pairings: ['Jamón ibérico', 'Paella', 'Cordero asado', 'Queso manchego'],
    characteristics: 'Cuerpo medio, taninos moderados, notas de cereza y cuero'
  }
};

// Importar red neuronal
import { NeuralNetGPU } from './neuralnet-gpu.js';
import { grapeCharacteristics } from './neuralnet.js';

// WebGPU Handler
class WinePairingWebGPU {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.supported = false;
    this.neuralNet = null;
  }

  async initialize() {
    if (!navigator.gpu) {
      console.warn('WebGPU no soportado en este navegador');
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        console.warn('No se pudo obtener adaptador WebGPU');
        return false;
      }

      this.device = await this.adapter.requestDevice();
      this.supported = true;
      
      // Inicializar red neuronal
      this.neuralNet = new NeuralNetGPU(this.device);
      await this.neuralNet.initialize();
      
      console.log('✅ WebGPU y Red Neuronal inicializados correctamente');
      return true;
    } catch (error) {
      console.error('Error inicializando WebGPU:', error);
      return false;
    }
  }

  // Procesar análisis de blend (mezcla) de vinos con IA
  async analyzeWineBlend(blend, year) {
    const yearValue = parseInt(year) || 2020;

    // Validar que todas las cepas existen
    for (const item of blend) {
      const grapeNormalized = item.grape.toLowerCase().trim();
      if (!grapeCharacteristics[grapeNormalized]) {
        return {
          success: false,
          message: `No se encontró información para la cepa: ${item.grape}`
        };
      }
    }

    try {
      // Si es una sola cepa, usar método simple
      if (blend.length === 1) {
        return this.analyzeWine(blend[0].grape, yearValue);
      }

      // Para blends, combinar predicciones según porcentajes
      const predictions = [];
      
      for (const item of blend) {
        const grapeNormalized = item.grape.toLowerCase().trim();
        const aiPrediction = this.neuralNet 
          ? await this.neuralNet.predict(grapeNormalized, yearValue)
          : this.neuralNet?.fallbackPredict(grapeNormalized, yearValue);
        
        if (aiPrediction) {
          predictions.push({
            ...aiPrediction,
            percentage: item.percentage
          });
        }
      }

      if (predictions.length === 0) {
        return this.fallbackAnalysis(blend[0].grape, year);
      }

      // Combinar predicciones ponderadas por porcentaje
      const combinedPredictions = this.combinePredictions(predictions);

      // Calcular métricas de análisis
      const currentYear = new Date().getFullYear();
      const age = currentYear - yearValue;
      const analysis = {
        quality: Math.min(100, Math.round(age * 5 + Math.random() * 30 + 50)),
        complexity: Math.round(75 + Math.random() * 20), // Blends son más complejos
        aging: Math.max(0, Math.round(20 - age + Math.random() * 10)),
        score: Math.round(80 + Math.random() * 15)
      };

      // Construir nombre del vino
      const wineName = blend.length === 1
        ? blend[0].grape.charAt(0).toUpperCase() + blend[0].grape.slice(1)
        : blend.map(b => `${b.grape.charAt(0).toUpperCase() + b.grape.slice(1)} (${b.percentage}%)`).join(' + ');

      // Obtener información de la cepa principal
      const mainGrape = blend[0].grape.toLowerCase().trim();
      const wineInfo = winePairingDatabase[mainGrape];

      let ageComment = '';
      if (age < 2) {
        ageComment = 'Un vino joven y fresco, ideal para disfrutar ahora.';
      } else if (age < 5) {
        ageComment = 'Un vino en su punto óptimo de consumo.';
      } else if (age < 10) {
        ageComment = 'Un vino con buena evolución y complejidad.';
      } else {
        ageComment = 'Un vino maduro con gran carácter.';
      }

      const blendDescription = blend.length > 1
        ? `Blend de ${blend.length} cepas que combina las características de cada varietal.`
        : wineInfo.description;

      // Convertir predicciones a formato con porcentajes
      const aiPairings = combinedPredictions.predictions.map(p => 
        `${p.category} (${Math.round(p.score * 100)}%)`
      );

      return {
        success: true,
        grape: wineName,
        year: yearValue,
        age: age,
        ageComment: ageComment,
        description: blendDescription,
        characteristics: blend.length > 1 
          ? 'Blend complejo con múltiples capas de sabor y aroma'
          : wineInfo.characteristics,
        pairings: aiPairings,
        aiPredictions: combinedPredictions.predictions,
        analysis: analysis,
        webGPUProcessed: predictions[0]?.usedGPU || false,
        aiPowered: true,
        isBlend: blend.length > 1,
        blendComponents: blend
      };

    } catch (error) {
      console.error('Error en análisis de blend:', error);
      return this.fallbackAnalysis(blend[0].grape, year);
    }
  }

  // Combinar predicciones de múltiples cepas según sus porcentajes
  combinePredictions(predictions) {
    // Crear un mapa para acumular scores por categoría
    const categoryScores = {};
    
    predictions.forEach(pred => {
      const weight = pred.percentage / 100;
      pred.predictions.forEach(p => {
        if (!categoryScores[p.category]) {
          categoryScores[p.category] = 0;
        }
        categoryScores[p.category] += p.score * weight;
      });
    });

    // Convertir a array y ordenar
    const combinedPredictions = Object.entries(categoryScores)
      .map(([category, score]) => ({ category, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    // Normalizar scores para que sumen aproximadamente 1
    const totalScore = combinedPredictions.reduce((sum, p) => sum + p.score, 0);
    combinedPredictions.forEach(p => {
      p.score = p.score / totalScore;
    });

    return {
      predictions: combinedPredictions,
      usedGPU: predictions[0]?.usedGPU || false
    };
  }

  // Procesar análisis de vino con IA (Red Neuronal en WebGPU) - varietal único
  async analyzeWine(grape, year) {
    const grapeNormalized = grape.toLowerCase().trim();
    const yearValue = parseInt(year) || 2020;

    // Verificar que la cepa existe
    if (!grapeCharacteristics[grapeNormalized]) {
      return {
        success: false,
        message: 'No se encontró información para esta cepa de uva. Intenta con: Malbec, Cabernet, Chardonnay, etc.'
      };
    }

    try {
      // Usar red neuronal para predecir maridajes
      const aiPrediction = this.neuralNet 
        ? await this.neuralNet.predict(grapeNormalized, yearValue)
        : this.neuralNet?.fallbackPredict(grapeNormalized, yearValue);

      if (!aiPrediction) {
        return this.fallbackAnalysis(grape, year);
      }
      
      // Calcular métricas de análisis
      const currentYear = new Date().getFullYear();
      const age = currentYear - yearValue;
      const analysis = {
        quality: Math.min(100, Math.round(age * 5 + Math.random() * 30 + 50)),
        complexity: Math.round(70 + Math.random() * 25),
        aging: Math.max(0, Math.round(20 - age + Math.random() * 10)),
        score: Math.round(75 + Math.random() * 20)
      };

      // Convertir predicciones de IA a formato de maridaje
      const aiPairings = aiPrediction.predictions.map(p => 
        `${p.category} (${Math.round(p.score * 100)}%)`
      );

      // Obtener información detallada del vino
      const wineInfo = winePairingDatabase[grapeNormalized];
      let ageComment = '';
      
      if (age < 2) {
        ageComment = 'Un vino joven y fresco, ideal para disfrutar ahora.';
      } else if (age < 5) {
        ageComment = 'Un vino en su punto óptimo de consumo.';
      } else if (age < 10) {
        ageComment = 'Un vino con buena evolución y complejidad.';
      } else {
        ageComment = 'Un vino maduro con gran carácter.';
      }

      return {
        success: true,
        grape: grapeNormalized.charAt(0).toUpperCase() + grapeNormalized.slice(1),
        year: yearValue,
        age: age,
        ageComment: ageComment,
        description: wineInfo.description,
        characteristics: wineInfo.characteristics,
        pairings: aiPairings, // Usar predicciones de IA
        aiPredictions: aiPrediction.predictions, // Incluir scores detallados
        analysis: analysis,
        webGPUProcessed: aiPrediction.usedGPU,
        aiPowered: true
      };

    } catch (error) {
      console.error('Error en análisis de IA:', error);
      return this.fallbackAnalysis(grape, year);
    }
  }

  fallbackAnalysis(grape, year) {
    const grapeNormalized = grape.toLowerCase().trim();
    const yearValue = parseInt(year) || 2020;
    
    // Análisis simple sin WebGPU
    const currentYear = new Date().getFullYear();
    const age = currentYear - yearValue;
    
    const analysis = {
      quality: Math.min(100, age * 5 + Math.random() * 30),
      complexity: Math.random() * 100,
      aging: Math.max(0, 20 - age),
      score: 50 + Math.random() * 40
    };

    return this.getWinePairing(grapeNormalized, yearValue, analysis);
  }
}

// Instancia global
const wineAnalyzer = new WinePairingWebGPU();

export { wineAnalyzer, WinePairingWebGPU };
