import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import { ScaleResponse } from "./types";

const App: React.FC = () => {
  const [scales, setScales] = useState<string[]>([]);
  const [selectedScale, setSelectedScale] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [annotations, setSaveAnnotation] = useState<boolean>(true);

  useEffect(() => {
    const connectToBackend = async () => {
      try {
        // Fetch available scales and the current scale
        const scaleResponse = await axios.get<ScaleResponse>("http://localhost:4000/scales");
        const { availableScales, currentScale } = scaleResponse.data;
        setScales(availableScales);
        setSelectedScale(currentScale);
      } catch (err) {
        setError("Failed to connect to the backend.");
        console.error("Backend connection error:", err);
      } finally {
        setLoading(false); // Hide loading spinner
      }
    };

    connectToBackend();
  }, []);

  const handleScaleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedScale(event.target.value);
  };

  const updateScale = async () => {
    try {
      const response = await axios.post("http://localhost:4000/update_scale", { scale: selectedScale });
      alert(response.data.message || "Scale updated successfully!");
    } catch (err) {
      alert("Failed to update scale.");
      console.error("Error updating scale:", err);
    }
  };

  const saveAnnotations = async () => {
    try {
      const response = await axios.post("http://localhost:4000/save_annotations", {
        message: "Save the current annotations",
      });
      alert(annotations ? "Annotations Saved Successfully!" : "Updated Annotations Successfully!");
      setSaveAnnotation(!annotations)
      
    } catch (err) {
      console.error("Error saving annotations:", err);
      alert("Failed to save annotations.");
    }
  };

  if (loading) {
    return <div>Loading...</div>; // Loading indicator
  }

  if (error) {
    return <div>Error: {error}</div>; // Error message
  }

  return (
    <div className="App">
      <h1>Guitar Neck Note Controller</h1>
      {/* <div className="video-container"> */}
        {/* Embed the video feed */}
        <img
          id="video-feed"
          src="http://localhost:4000/video_feed"
          alt="Video Stream"
          className="video-container"
        />
      {/* </div> */}
      <div className="controls">
        <label htmlFor="scale-select">Select Scale:</label>
        <select
          id="scale-select"
          value={selectedScale}
          onChange={handleScaleChange}
        >
        {scales.map((scale) => (
          <option key={scale} value={scale}>
              {scale}
            </option>
          ))}
        </select>
        
        <button onClick={updateScale}>Update Scale</button>
        <button onClick={saveAnnotations}> {annotations ? "Save Annotations" : "Update Annotations"}
        </button>
      </div>
    </div>
  );
};

export default App;