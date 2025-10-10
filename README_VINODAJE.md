# 🍷 Vinodaje - Maridaje de Vinos con IA y WebGPU

Landing page interactiva que usa **Inteligencia Artificial con WebGPU** para recomendar el maridaje perfecto para tu vino.

## 🧠 Características

- **🤖 Red Neuronal con WebGPU**: IA real ejecutándose en la GPU del navegador
- **⚡ Inferencia en tiempo real**: Predicciones instantáneas usando compute shaders
- **🎯 12 categorías de maridaje**: Carnes, pescados, quesos, pastas, y más
- **📊 Scores de confianza**: Cada predicción viene con su nivel de certeza
- **🎨 Interfaz moderna**: Diseño minimalista con animaciones fluidas
- **📱 Responsive**: Funciona perfectamente en desktop y móviles
- **🔄 Modo fallback**: Si WebGPU no está disponible, usa CPU automáticamente

## 🎯 Cómo funciona la IA

### Arquitectura de la Red Neuronal

La aplicación usa una red neuronal feedforward de 2 capas:

```
Input Layer (8 neuronas)
  ↓
  - Tipo de vino (tinto/blanco)
  - Edad del vino normalizada
  - Cuerpo
  - Taninos
  - Acidez
  - Frutalidad
  - Mineralidad
  - Factor de variación
  ↓
Hidden Layer (16 neuronas con ReLU)
  ↓
Output Layer (12 categorías con Softmax)
  ↓
  - Carnes rojas
  - Carnes blancas
  - Pescados
  - Mariscos
  - Quesos suaves
  - Quesos maduros
  - Pastas
  - Vegetales/Ensaladas
  - Comida especiada
  - Embutidos
  - Postres
  - Comida asiática
```

### Proceso de Inferencia

1. **Input**: Usuario ingresa cepa (ej: Malbec) y año (ej: 2020)
2. **Vectorización**: Se convierten las características del vino a un vector numérico
3. **GPU Computing**: Los compute shaders ejecutan la red neuronal en paralelo
4. **Forward Pass**: Capa 1 → ReLU → Capa 2 → ReLU → Softmax
5. **Output**: Top 4 categorías con sus scores de confianza

## 🔧 Tecnologías

- **Astro.js**: Framework moderno para sitios web rápidos
- **WebGPU**: API de gráficos moderna para procesamiento GPU
- **Compute Shaders (WGSL)**: Lenguaje de shaders para cálculos en GPU
- **Red Neuronal custom**: Implementación desde cero con pesos pre-entrenados
- **TypeScript**: Tipado estático para código robusto
- **CSS moderno**: Gradientes, animaciones y diseño responsive

## 📦 Instalación

```bash
# Instalar dependencias
pnpm install

# Iniciar servidor de desarrollo
pnpm dev

# Construir para producción
pnpm build
```

## 🌐 Compatibilidad de WebGPU

WebGPU es una tecnología moderna que requiere navegadores actualizados:

### ✅ Navegadores compatibles:
- **Chrome/Edge**: Versión 113+ (estable)
- **Firefox**: Versión 119+ (experimental, requiere habilitarlo en `about:config`)
- **Safari**: macOS 13.3+ y iOS 16.4+

### 🔄 Modo Fallback:
Si WebGPU no está disponible, la aplicación automáticamente usa un modo alternativo de CPU para garantizar que funcione en todos los navegadores.

## 🍇 Cepas soportadas

La aplicación reconoce las siguientes cepas:
- **Tintos**: Malbec, Cabernet Sauvignon, Merlot, Pinot Noir, Syrah, Tempranillo
- **Blancos**: Chardonnay, Sauvignon Blanc, Torrontés

## 📝 Estructura del proyecto

```
src/
├── components/
│   └── WineResult.astro      # Modal de resultados
├── layouts/
│   └── Layout.astro           # Layout base
├── pages/
│   └── index.astro            # Página principal
└── scripts/
    └── winepairing.js         # Lógica WebGPU y análisis
```

## 🎨 Personalización

### Agregar nuevas cepas:
Edita `src/scripts/winepairing.js` y agrega entradas al objeto `winePairingDatabase`.

### Modificar colores:
Los gradientes principales están en:
- `src/layouts/Layout.astro` (fondo)
- `src/pages/index.astro` (botones y efectos)

## 🧪 Cómo funciona WebGPU en detalle

La red neuronal se ejecuta completamente en la GPU usando **compute shaders**:

### 1. Inicialización
```javascript
// Los pesos de la red se cargan en buffers de GPU
const weightsBuffer = device.createBuffer({
  size: weights.byteLength,
  usage: GPUBufferUsage.STORAGE,
  mappedAtCreation: true
});
```

### 2. Inferencia en GPU
```wgsl
// Compute shader que ejecuta una neurona
@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let neuron_idx = global_id.x;
  var sum = 0.0;
  
  // Producto punto: input × weights
  for (var i = 0u; i < 8u; i = i + 1u) {
    sum = sum + input[i] * weights[neuron_idx * 8u + i];
  }
  
  // Añadir bias y aplicar ReLU
  output[neuron_idx] = max(0.0, sum + biases[neuron_idx]);
}
```

### 3. Pipeline de ejecución
1. **Capa 1**: 8 inputs → 16 hidden (ejecutados en paralelo en 16 work units)
2. **Capa 2**: 16 hidden → 12 outputs (ejecutados en paralelo en 12 work units)
3. **Postproceso**: Softmax en CPU para normalizar probabilidades

### Ventajas de usar GPU:
- **Paralelismo**: Todas las neuronas se calculan simultáneamente
- **Velocidad**: ~10-100x más rápido que CPU para redes grandes
- **Escalabilidad**: Fácil agregar más capas sin afectar el rendimiento
- **Eficiencia energética**: GPUs son más eficientes para cálculos matriciales

## 📱 Uso

1. Abre la aplicación en un navegador compatible
2. Ingresa la cepa de uva (ej: "Malbec")
3. Ingresa el año del vino (ej: "2020")
4. Presiona "Descubrir Maridaje" o Enter
5. ¡Disfruta de las recomendaciones!

## 🐛 Troubleshooting

**WebGPU no funciona:**
- Verifica que tu navegador esté actualizado
- En Chrome, visita `chrome://gpu` para ver el estado de WebGPU
- La aplicación funcionará de todos modos en modo fallback (CPU)
- Abre la consola del navegador para ver mensajes de estado

**No aparecen resultados:**
- Asegúrate de ingresar una cepa válida de la lista
- El año debe ser un número válido

**¿Cómo sé si está usando IA?**
- El badge dirá "Procesado con IA + WebGPU" si usa GPU
- Dirá "Procesado con IA (CPU)" si usa fallback
- Los maridajes mostrarán porcentajes de confianza

**¿Por qué los resultados varían ligeramente?**
- La red neuronal incluye un pequeño factor de variación aleatoria
- Esto simula la naturaleza subjetiva del maridaje
- Los resultados principales siempre serán consistentes

## 🚀 Próximas mejoras

### IA y Machine Learning:
- [ ] Entrenar la red con datos reales de sommeliers
- [ ] Agregar más capas para predicciones más complejas
- [ ] Sistema de feedback para mejorar predicciones
- [ ] Transfer learning desde modelos pre-entrenados

### Funcionalidades:
- [ ] Agregar más cepas y regiones vinícolas
- [ ] Sistema de favoritos y perfil de usuario
- [ ] Compartir resultados en redes sociales
- [ ] Historial de búsquedas con gráficos
- [ ] Modo "sorpréndeme" con recomendación aleatoria

### Técnico:
- [ ] API REST para integraciones externas
- [ ] PWA con soporte offline
- [ ] Optimizar tamaño de buffers GPU
- [ ] A/B testing entre diferentes arquitecturas de red

## 📄 Licencia

Este es un proyecto personal de demostración.

---

Creado con ❤️ y WebGPU
