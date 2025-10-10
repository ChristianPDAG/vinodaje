// Red Neuronal para Maridaje de Vinos con WebGPU
// Arquitectura: Input (8) -> Hidden (16) -> Output (12 categorías de comida)

// Pesos pre-entrenados basados en conocimiento enológico
// Estos pesos fueron "entrenados" con patrones de maridaje tradicionales
export const neuralNetWeights = {
  // Capa 1: Input (8) -> Hidden (16)
  layer1: {
    weights: new Float32Array([
      // Neurona 0 - Detecta tintos robustos
      0.8, 0.3, 0.1, 0.9, 0.7, -0.3, -0.2, 0.4,
      // Neurona 1 - Detecta tintos suaves
      0.6, 0.2, 0.1, 0.5, 0.4, -0.1, -0.1, 0.3,
      // Neurona 2 - Detecta blancos frescos
      -0.7, -0.2, 0.3, -0.5, -0.6, 0.8, 0.7, 0.2,
      // Neurona 3 - Detecta blancos con cuerpo
      -0.5, -0.1, 0.2, -0.3, -0.4, 0.6, 0.5, 0.1,
      // Neurona 4 - Detecta vinos jóvenes
      0.2, 0.8, 0.6, 0.1, 0.2, 0.1, 0.2, 0.5,
      // Neurona 5 - Detecta vinos añejos
      0.3, -0.6, -0.5, 0.4, 0.5, 0.2, 0.1, 0.6,
      // Neurona 6 - Detecta alta acidez
      0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.9, 0.3,
      // Neurona 7 - Detecta taninos altos
      0.9, 0.1, 0.1, 0.8, 0.9, -0.2, -0.1, 0.2,
      // Neurona 8 - Complejidad aromática
      0.4, 0.3, 0.5, 0.5, 0.4, 0.6, 0.5, 0.8,
      // Neurona 9 - Potencial de guarda
      0.5, -0.4, -0.3, 0.6, 0.7, 0.3, 0.2, 0.7,
      // Neurona 10 - Frutalidad
      0.3, 0.5, 0.4, 0.4, 0.3, 0.5, 0.4, 0.6,
      // Neurona 11 - Especiado
      0.7, 0.2, 0.2, 0.6, 0.5, 0.2, 0.1, 0.5,
      // Neurona 12 - Mineral
      -0.4, 0.3, 0.2, -0.3, -0.2, 0.7, 0.8, 0.4,
      // Neurona 13 - Dulzor residual
      -0.2, 0.4, 0.3, -0.1, -0.1, 0.5, 0.3, 0.3,
      // Neurona 14 - Estructura
      0.8, -0.2, -0.1, 0.7, 0.8, 0.2, 0.1, 0.4,
      // Neurona 15 - Versatilidad
      0.4, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5
    ]),
    biases: new Float32Array([
      -0.1, -0.05, 0.05, 0.0, 0.1, -0.15, 0.0, -0.1,
      0.0, -0.1, 0.05, -0.05, 0.1, 0.05, -0.1, 0.0
    ])
  },
  // Capa 2: Hidden (16) -> Output (12 categorías de comida)
  layer2: {
    weights: new Float32Array([
      // Output 0: Carnes rojas
      0.9, 0.6, -0.5, -0.3, 0.2, 0.7, -0.1, 0.8, 0.3, 0.6, 0.4, 0.7, -0.2, -0.1, 0.8, 0.4,
      // Output 1: Carnes blancas
      0.4, 0.7, -0.2, 0.3, 0.5, 0.2, 0.1, 0.3, 0.5, 0.3, 0.6, 0.4, 0.1, 0.2, 0.4, 0.7,
      // Output 2: Pescados
      -0.6, -0.3, 0.8, 0.7, 0.4, -0.2, 0.6, -0.4, 0.5, -0.1, 0.6, 0.1, 0.7, 0.3, -0.3, 0.5,
      // Output 3: Mariscos
      -0.7, -0.4, 0.9, 0.8, 0.5, -0.3, 0.8, -0.5, 0.4, -0.2, 0.5, 0.0, 0.8, 0.2, -0.4, 0.4,
      // Output 4: Quesos suaves
      0.2, 0.5, 0.3, 0.6, 0.3, 0.1, 0.2, 0.1, 0.6, 0.2, 0.5, 0.2, 0.3, 0.4, 0.2, 0.6,
      // Output 5: Quesos maduros
      0.7, 0.4, -0.3, -0.1, 0.1, 0.8, 0.0, 0.6, 0.4, 0.7, 0.3, 0.5, -0.1, 0.1, 0.7, 0.3,
      // Output 6: Pastas
      0.5, 0.6, 0.1, 0.2, 0.4, 0.3, 0.1, 0.4, 0.5, 0.3, 0.5, 0.4, 0.2, 0.3, 0.4, 0.6,
      // Output 7: Vegetales/Ensaladas
      -0.4, 0.2, 0.7, 0.6, 0.6, -0.2, 0.7, -0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.4, -0.2, 0.6,
      // Output 8: Comida especiada
      0.6, 0.3, 0.2, 0.1, 0.2, 0.4, 0.3, 0.5, 0.7, 0.4, 0.5, 0.8, 0.3, 0.3, 0.5, 0.4,
      // Output 9: Embutidos
      0.5, 0.4, -0.1, 0.0, 0.3, 0.5, 0.1, 0.4, 0.4, 0.5, 0.4, 0.6, 0.0, 0.2, 0.5, 0.5,
      // Output 10: Postres
      -0.3, 0.3, 0.5, 0.4, 0.4, -0.1, 0.4, -0.2, 0.6, 0.1, 0.6, 0.2, 0.4, 0.7, -0.1, 0.5,
      // Output 11: Comida asiática
      0.1, 0.4, 0.6, 0.5, 0.5, 0.1, 0.5, 0.0, 0.7, 0.2, 0.7, 0.4, 0.5, 0.4, 0.1, 0.6
    ]),
    biases: new Float32Array([
      0.1, 0.05, 0.0, -0.05, 0.0, 0.1, 0.05, 0.0, 0.0, 0.05, -0.1, 0.0
    ])
  }
};

// Mapeo de categorías de salida a nombres
export const foodCategories = [
  'Carnes rojas',
  'Carnes blancas', 
  'Pescados',
  'Mariscos',
  'Quesos suaves',
  'Quesos maduros',
  'Pastas',
  'Vegetales/Ensaladas',
  'Comida especiada',
  'Embutidos',
  'Postres',
  'Comida asiática'
];

// Características de cada cepa (para el input de la red)
export const grapeCharacteristics = {
  'malbec': {
    type: 1.0,        // Tinto robusto
    body: 0.9,        // Cuerpo completo
    tannins: 0.7,     // Taninos medios-altos
    acidity: 0.5,     // Acidez media
    fruitiness: 0.8,  // Muy frutal
    minerality: 0.2   // Baja mineralidad
  },
  'cabernet sauvignon': {
    type: 1.0,
    body: 1.0,
    tannins: 0.9,
    acidity: 0.6,
    fruitiness: 0.7,
    minerality: 0.3
  },
  'cabernet': {
    type: 1.0,
    body: 1.0,
    tannins: 0.9,
    acidity: 0.6,
    fruitiness: 0.7,
    minerality: 0.3
  },
  'merlot': {
    type: 0.8,
    body: 0.7,
    tannins: 0.5,
    acidity: 0.5,
    fruitiness: 0.8,
    minerality: 0.2
  },
  'pinot noir': {
    type: 0.6,
    body: 0.5,
    tannins: 0.4,
    acidity: 0.7,
    fruitiness: 0.9,
    minerality: 0.5
  },
  'syrah': {
    type: 1.0,
    body: 0.9,
    tannins: 0.8,
    acidity: 0.6,
    fruitiness: 0.7,
    minerality: 0.3
  },
  'chardonnay': {
    type: -1.0,       // Blanco
    body: 0.7,
    tannins: 0.0,
    acidity: 0.6,
    fruitiness: 0.6,
    minerality: 0.6
  },
  'sauvignon blanc': {
    type: -1.0,
    body: 0.4,
    tannins: 0.0,
    acidity: 0.9,
    fruitiness: 0.8,
    minerality: 0.7
  },
  'torrontés': {
    type: -1.0,
    body: 0.5,
    tannins: 0.0,
    acidity: 0.7,
    fruitiness: 0.9,
    minerality: 0.5
  },
  'tempranillo': {
    type: 0.9,
    body: 0.8,
    tannins: 0.7,
    acidity: 0.6,
    fruitiness: 0.7,
    minerality: 0.4
  }
};

// Crear vector de entrada para la red neuronal
export function createInputVector(grape, year) {
  const characteristics = grapeCharacteristics[grape.toLowerCase()] || grapeCharacteristics['malbec'];
  const currentYear = 2025;
  const age = currentYear - parseInt(year);
  
  // Normalizar edad (0 = muy joven, 1 = muy añejo)
  const normalizedAge = Math.min(age / 20, 1.0);
  
  // Vector de 8 características
  return new Float32Array([
    characteristics.type,
    normalizedAge,
    characteristics.body,
    characteristics.tannins,
    characteristics.acidity,
    characteristics.fruitiness,
    characteristics.minerality,
    Math.random() * 0.1 // Factor de variación pequeño
  ]);
}

// Función de activación ReLU
export function relu(x) {
  return Math.max(0, x);
}

// Función softmax para normalizar scores de salida
export function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}
