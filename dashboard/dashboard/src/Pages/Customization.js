import React, { useState } from 'react';
import './Customization.css'; // Import your CSS file for styling

export default function Customization() {
  // Define the initial state for AI Assistant parameters
  const [aiAssistantParameters, setAiAssistantParameters] = useState({
    responseLatency: '',
    vocabularySize: '',
    contextWindow: '',
    confidenceThreshold: '',
  });

  // Function to handle AI Assistant parameter changes
  const handleAiAssistantParameterChange = (parameterName, value) => {
    setAiAssistantParameters((prevParameters) => ({
      ...prevParameters,
      [parameterName]: value,
    }));
  };

// Define the initial state for AI Assistant parameters
    const [OCRParameters, setOCRParameters] = useState({
        responseLatency: '',
        vocabularySize: '',
        contextWindow: '',
        confidenceThreshold: '',
        });

    // Function to handle AI Assistant parameter changes
    const handleOCRParameterChange = (parameterName, value) => {
    setOCRParameters((prevParameters) => ({
        ...prevParameters,
        [parameterName]: value,
    }));
    };

    // Define the initial state for AI Assistant parameters
  const [objectFinderParameters, setobjectFinderParameters] = useState({
    responseLatency: '',
    vocabularySize: '',
    contextWindow: '',
    confidenceThreshold: '',
  });

  // Function to handle AI Assistant parameter changes
  const handleobjectFinderParameterChange = (parameterName, value) => {
    setobjectFinderParameters((prevParameters) => ({
      ...prevParameters,
      [parameterName]: value,
    }));
  };

  const [objectDetectionParameters, setObjectDetectionParameters] = useState({
    responseLatency: '',
    vocabularySize: '',
    contextWindow: '',
    confidenceThreshold: '',
  });

  // Function to handle AI Assistant parameter changes
  const handleObjectDetectionParameterChange = (parameterName, value) => {
    setObjectDetectionParameters((prevParameters) => ({
      ...prevParameters,
      [parameterName]: value,
    }));
  };

  const [navigationParameters, setNavigationParameters] = useState({
    responseLatency: '',
    vocabularySize: '',
    contextWindow: '',
    confidenceThreshold: '',
  });

  // Function to handle AI Assistant parameter changes
  const handleNavigationParameterChange = (parameterName, value) => {
    setNavigationParameters((prevParameters) => ({
      ...prevParameters,
      [parameterName]: value,
    }));
  };

  const [environmentParameters, setEnvironmentParameters] = useState({
    responseLatency: '',
    vocabularySize: '',
    contextWindow: '',
    confidenceThreshold: '',
  });

  // Function to handle AI Assistant parameter changes
  const handleEnvironmentParameterChange = (parameterName, value) => {
    setEnvironmentParameters((prevParameters) => ({
      ...prevParameters,
      [parameterName]: value,
    }));
  };

  return (
    <div className="customization-container">
      <h1>Customization</h1>
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>Parameters</th>
          </tr>
        </thead>
        <tbody>
            {/* AI Assistant */}
          <tr>
            <td>AI Assistant</td>
            <td>
              <ul className="parameter-list">
                <li>
                  <label>
                    Response Latency:
                    <select
                      value={aiAssistantParameters.responseLatency}
                      onChange={(e) =>
                        handleAiAssistantParameterChange(
                          'responseLatency',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="100ms">100ms</option>
                      <option value="500ms">500ms</option>
                      <option value="1s">1s</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Vocabulary Size:
                    <select
                      value={aiAssistantParameters.vocabularySize}
                      onChange={(e) =>
                        handleAiAssistantParameterChange(
                          'vocabularySize',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="50">50</option>
                      <option value="100">100</option>
                      <option value="200">200</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Context Window:
                    <select
                      value={aiAssistantParameters.contextWindow}
                      onChange={(e) =>
                        handleAiAssistantParameterChange(
                          'contextWindow',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">2min</option>
                      <option value="5min">5min</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Confidence Threshold:
                    <input
                      type="text"
                      value={aiAssistantParameters.confidenceThreshold}
                      onChange={(e) =>
                        handleAiAssistantParameterChange(
                          'confidenceThreshold',
                          e.target.value
                        )
                      }
                    />
                  </label>
                </li>
              </ul>
            </td>
          </tr>
            {/* OCR */}
          <tr>
            <td>OCR</td>
            <td>
              <ul className="parameter-list">
                <li>
                  <label>
                    Summary Length:
                    <select
                      value={OCRParameters.responseLatency}
                      onChange={(e) =>
                        handleOCRParameterChange(
                          'responseLatency',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="100ms">50</option>
                      <option value="500ms">100</option>
                      <option value="1s">200</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Keyword Importance:
                    <select
                      value={OCRParameters.vocabularySize}
                      onChange={(e) =>
                        handleOCRParameterChange(
                          'vocabularySize',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="50">Low</option>
                      <option value="100">Medium</option>
                      <option value="200">High</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Font Sensitivity:
                    <select
                      value={OCRParameters.contextWindow}
                      onChange={(e) =>
                        handleOCRParameterChange(
                          'contextWindow',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">Low</option>
                      <option value="5min">Medium</option>
                      <option value="200">High</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Summarization Algorithm:
                    <select
                      value={OCRParameters.confidenceThreshold}
                      onChange={(e) =>
                        handleOCRParameterChange(
                          'confidenceThreshold',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="Extractive">Extractive</option>
                      <option value="Abstractive">Abstractive</option>
                      <option value="Hybrid">Hybrid</option>
                    </select>
                  </label>
                </li>
              </ul>
            </td>
          </tr>
            {/* Object Finder */}
          <tr>
            <td>Object Finder</td>
            <td>
              <ul className="parameter-list">
                <li>
                  <label>
                    Search Depth:
                    <select
                      value={objectFinderParameters.responseLatency}
                      onChange={(e) =>
                        handleobjectFinderParameterChange(
                          'responseLatency',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="100ms">1m</option>
                      <option value="500ms">2m</option>
                      <option value="1s">5m</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Precision Vs Recall:
                    <select
                      value={objectFinderParameters.vocabularySize}
                      onChange={(e) =>
                        handleobjectFinderParameterChange(
                          'vocabularySize',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="50">Precision</option>
                      <option value="100">Recall</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Search Speed:
                    <select
                      value={objectFinderParameters.contextWindow}
                      onChange={(e) =>
                        handleobjectFinderParameterChange(
                          'contextWindow',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">Low</option>
                      <option value="5min">Medium</option>
                      <option value="10min">High</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Object Complexity:
                    <select
                      value={objectFinderParameters.confidenceThreshold}
                      onChange={(e) =>
                        handleobjectFinderParameterChange(
                          'confidenceThreshold',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">Low</option>
                      <option value="5min">Medium</option>
                      <option value="10min">High</option>
                    </select>
                  </label>
                </li>
              </ul>
            </td>
          </tr>
            {/* Object Detection */}
          <tr>
            <td>Object Detection</td>
            <td>
              <ul className="parameter-list">
                <li>
                  <label>
                    Building Box Dimensions:
                    <input
                      type="text"
                      value={objectDetectionParameters.responseLatency}
                      onChange={(e) =>
                        handleObjectDetectionParameterChange(
                          'responseLatency',
                          e.target.value
                        )
                      }
                    />
                  </label>
                </li>
                <li>
                  <label>
                    Latency:
                    <select
                      value={objectDetectionParameters.vocabularySize}
                      onChange={(e) =>
                        handleObjectDetectionParameterChange(
                          'vocabularySize',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="50">100ms</option>
                      <option value="100">500ms</option>
                      <option value="200">1s</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Frame Rate:
                    <select
                      value={objectDetectionParameters.contextWindow}
                      onChange={(e) =>
                        handleObjectDetectionParameterChange(
                          'contextWindow',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">10FPS</option>
                      <option value="5min">30FPS</option>
                      <option value="10min">60PFS</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Confidence Threshold:
                    <input
                      type="text"
                      value={objectDetectionParameters.confidenceThreshold}
                      onChange={(e) =>
                        handleObjectDetectionParameterChange(
                          'confidenceThreshold',
                          e.target.value
                        )
                      }
                    />
                  </label>
                </li>
              </ul>
            </td>
          </tr>
            {/* Navigation */}
          <tr>
            <td>Navigation</td>
            <td>
              <ul className="parameter-list">
                <li>
                  <label>
                    Localization Accuracy:
                    <select
                      value={navigationParameters.responseLatency}
                      onChange={(e) =>
                        handleNavigationParameterChange(
                          'responseLatency',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="100ms">1m</option>
                      <option value="500ms">10m</option>
                      <option value="1s">20m</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Path Algorithm:
                    <select
                      value={navigationParameters.vocabularySize}
                      onChange={(e) =>
                        handleNavigationParameterChange(
                          'vocabularySize',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="50">Dijkstra</option>
                      <option value="100">A*</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Safety Margin:
                    <select
                      value={navigationParameters.contextWindow}
                      onChange={(e) =>
                        handleNavigationParameterChange(
                          'contextWindow',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">0.5m</option>
                      <option value="5min">1m</option>
                      <option value="10min">5m</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Speed vs Accuracy:
                    <select
                      value={navigationParameters.confidenceThreshold}
                      onChange={(e) =>
                        handleNavigationParameterChange(
                          'confidenceThreshold',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">Speed</option>
                      <option value="5min">Accuracy</option>
                    </select>
                  </label>
                </li>
              </ul>
            </td>
          </tr>
            {/* Environment Detection */}
          <tr>
            <td>Environment Detection</td>
            <td>
              <ul className="parameter-list">
                <li>
                  <label>
                    Spatial Resolution:
                    <select
                      value={environmentParameters.responseLatency}
                      onChange={(e) =>
                        handleEnvironmentParameterChange(
                          'responseLatency',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="100ms">100ms</option>
                      <option value="500ms">500ms</option>
                      <option value="1s">1s</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Noise Reduction:
                    <select
                      value={environmentParameters.vocabularySize}
                      onChange={(e) =>
                        handleEnvironmentParameterChange(
                          'vocabularySize',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="50">50</option>
                      <option value="100">100</option>
                      <option value="200">200</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Vibration Tolerance:
                    <select
                      value={environmentParameters.contextWindow}
                      onChange={(e) =>
                        handleEnvironmentParameterChange(
                          'contextWindow',
                          e.target.value
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="2min">2min</option>
                      <option value="5min">5min</option>
                    </select>
                  </label>
                </li>
                <li>
                  <label>
                    Confidence Threshold:
                    <input
                      type="text"
                      value={aiAssistantParameters.confidenceThreshold}
                      onChange={(e) =>
                        handleAiAssistantParameterChange(
                          'confidenceThreshold',
                          e.target.value
                        )
                      }
                    />
                  </label>
                </li>
              </ul>
            </td>
          </tr>

        </tbody>
      </table>
    </div>
  );
}
