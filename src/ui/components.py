"""
UI components for the Streamlit application.
"""
import time
import streamlit as st
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from config import config

# Import visualizations directly to avoid circular imports
from src.ui.visualizations import (
    plot_scores_history,
    plot_scores_gauge,
    highlight_misalignment_in_text,
    create_face_grid
)


def header_section():
    """
    Display the header section of the UI.
    """
    st.title("Conversation Misalignment Detection System")
    st.markdown("""
    This system detects misalignment and misunderstanding during conversations using both camera and speech inputs.
    It analyzes facial expressions and speech patterns to identify signs of confusion and misunderstanding.
    """)
    

def camera_section(components: Dict):
    """
    Display the camera section of the UI.
    
    Args:
        components: Dictionary of system components
    """
    st.header("Visual Misalignment")
    
    camera_manager = components["camera_manager"]
    face_detector = components["face_detector"]
    camera_processor = components["camera_processor"]
    
    # Get camera frames
    frames = camera_manager.get_frames()
    
    if not frames or all(frame[0] is None for frame in frames.values()):
        st.warning("No camera feed available")
        return
        
    # Display camera feeds with face landmarks
    if config.ui.show_facial_features:
        # Get latest camera scores
        camera_scores = camera_processor.get_all_latest_scores()
        
        # Process and display frames
        processed_frames = {}
        for person_name, (timestamp, frame) in frames.items():
            if frame is not None:
                # Detect face
                success, face_data = face_detector.detect_face(frame)
                
                # Draw landmarks if face detected
                if success:
                    frame_with_landmarks = face_detector.draw_face_landmarks(frame, face_data)
                    processed_frames[person_name] = frame_with_landmarks
                else:
                    processed_frames[person_name] = frame
                    
                # Add score overlay if available
                if person_name in camera_scores:
                    score, _, _ = camera_scores[person_name]
                    
                    # Draw score in top-left corner
                    cv2.putText(
                        processed_frames[person_name],
                        f"Score: {score}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255) if score > 50 else (0, 255, 0),
                        2
                    )
        
        # Display face grid
        st.image(create_face_grid(processed_frames), channels="BGR", use_column_width=True)
    else:
        # Display raw camera feeds
        st.image(create_face_grid(frames), channels="BGR", use_column_width=True)
    
    # Display camera misalignment scores
    cols = st.columns(len(frames))
    
    for i, person_name in enumerate(frames.keys()):
        with cols[i]:
            smoothed_score = camera_processor.get_smoothed_score(person_name)
            st.metric(
                f"{person_name} Visual Score", 
                f"{smoothed_score}",
                delta=None,
                delta_color="inverse"
            )
            
            if config.ui.show_raw_scores:
                # Show gauge chart
                plot_scores_gauge(
                    smoothed_score, 
                    f"{person_name} Visual Misalignment", 
                    [0, 20, 50, 80, 100]
                )
                
                # Show trend if available
                trend = camera_processor.get_score_trend(person_name)
                if trend != 'unknown':
                    st.caption(f"Trend: {trend}")
    
    # Show score history if configured
    if config.ui.show_history:
        # Get history for each person
        histories = {}
        for person_name in frames.keys():
            history = camera_processor.get_history(person_name, max_items=config.ui.chart_history_size)
            if history:
                # Extract timestamps and scores
                timestamps = [h[0] for h in history]
                scores = [h[1] for h in history]
                histories[person_name] = (timestamps, scores)
        
        if histories:
            st.subheader("Visual Misalignment History")
            plot_scores_history(histories, "Visual Misalignment Score")
            

def speech_section(components: Dict):
    """
    Display the speech section of the UI.
    
    Args:
        components: Dictionary of system components
    """
    st.header("Speech Misalignment")
    
    audio_recorder = components["audio_recorder"]
    speech_processor = components["speech_processor"]
    
    # Get audio device status
    device_status = audio_recorder.get_device_status()
    
    if not device_status or all(not status["is_recording"] for status in device_status.values()):
        st.warning("No audio recording available")
        return
    
    # Display active devices and their status
    active_devices = [name for name, status in device_status.items() if status["active"]]
    
    if active_devices:
        st.success(f"Active audio detected from: {', '.join(active_devices)}")
    else:
        st.info("No active speech detected")
    
    # Display speech misalignment scores
    cols = st.columns(len(device_status))
    
    for i, person_name in enumerate(device_status.keys()):
        with cols[i]:
            smoothed_score = speech_processor.get_smoothed_score(person_name)
            st.metric(
                f"{person_name} Speech Score", 
                f"{smoothed_score}",
                delta=None,
                delta_color="inverse"
            )
            
            if config.ui.show_raw_scores:
                # Show gauge chart
                plot_scores_gauge(
                    smoothed_score, 
                    f"{person_name} Speech Misalignment", 
                    [0, 20, 50, 80, 100]
                )
                
                # Show trend if available
                trend = speech_processor.get_score_trend(person_name)
                if trend != 'unknown':
                    st.caption(f"Trend: {trend}")
    
    # Show score history if configured
    if config.ui.show_history:
        # Get history for each person
        histories = {}
        for person_name in device_status.keys():
            history = speech_processor.get_history(person_name, max_items=config.ui.chart_history_size)
            if history:
                # Extract timestamps and scores
                timestamps = [h[0] for h in history]
                scores = [h[1] for h in history]
                histories[person_name] = (timestamps, scores)
        
        if histories:
            st.subheader("Speech Misalignment History")
            plot_scores_history(histories, "Speech Misalignment Score")
            

def combined_section(components: Dict):
    """
    Display the combined misalignment section of the UI.
    
    Args:
        components: Dictionary of system components
    """
    st.header("Combined Misalignment")
    
    combined_processor = components["combined_processor"]
    
    # Get all persons with scores
    all_scores = combined_processor.get_all_latest_scores()
    person_names = list(all_scores.keys())
    
    if not person_names:
        st.info("No misalignment data available yet")
        return
    
    # Display combined misalignment scores
    cols = st.columns(len(person_names))
    
    for i, person_name in enumerate(person_names):
        with cols[i]:
            smoothed_score = combined_processor.get_smoothed_score(person_name)
            
            # Determine alert color
            if smoothed_score >= 80:
                color = "🔴"  # Red for high misalignment
            elif smoothed_score >= 50:
                color = "🟠"  # Orange for moderate misalignment
            elif smoothed_score >= 20:
                color = "🟡"  # Yellow for mild misalignment
            else:
                color = "🟢"  # Green for low misalignment
                
            st.metric(
                f"{person_name} Combined Score", 
                f"{color} {smoothed_score}",
                delta=None
            )
            
            # Show description
            description = combined_processor.get_misalignment_description(smoothed_score)
            st.caption(description)
            
            if config.ui.show_combined_score:
                # Show gauge chart
                plot_scores_gauge(
                    smoothed_score, 
                    f"{person_name} Combined Misalignment", 
                    [0, 20, 50, 80, 100]
                )
    
    # Show detailed breakdown for the first person (expandable)
    if person_names:
        with st.expander("View Score Breakdown Details"):
            person_name = person_names[0]
            score, details, timestamp = all_scores[person_name]
            
            # Extract component scores
            camera_score = details.get("camera_score", 0)
            speech_score = details.get("speech_score", 0)
            
            # Display component scores
            cols = st.columns(2)
            with cols[0]:
                st.metric("Visual Score", camera_score)
            with cols[1]:
                st.metric("Speech Score", speech_score)
                
            # Display weights
            weights = details.get("weights", {})
            st.caption(f"Weight ratio: Visual {weights.get('camera', 0.5):.1f} : Speech {weights.get('speech', 0.5):.1f}")
            
            # Display LLM analysis if available
            llm_analysis = details.get("llm_analysis")
            if llm_analysis:
                st.subheader("AI Analysis")
                
                explanation = llm_analysis.get("explanation", "")
                likely_cause = llm_analysis.get("likely_cause", "")
                suggestions = llm_analysis.get("suggestions", [])
                
                if explanation:
                    st.write(f"**Analysis:** {explanation}")
                if likely_cause:
                    st.write(f"**Likely cause:** {likely_cause}")
                if suggestions:
                    st.write("**Suggestions:**")
                    for suggestion in suggestions:
                        st.write(f"- {suggestion}")
    
    # Show score history if configured
    if config.ui.show_history:
        # Get history for each person
        histories = {}
        for person_name in person_names:
            history = combined_processor.get_history(person_name, max_items=config.ui.chart_history_size)
            if history:
                # Extract timestamps and scores
                timestamps = [h[0] for h in history]
                scores = [h[1] for h in history]
                histories[person_name] = (timestamps, scores)
        
        if histories:
            st.subheader("Combined Misalignment History")
            plot_scores_history(histories, "Combined Misalignment Score")
            

def transcript_section(components: Dict):
    """
    Display the transcript section of the UI.
    
    Args:
        components: Dictionary of system components
    """
    st.header("Conversation Transcript")
    
    speech_processor = components["speech_processor"]
    speech_analyzer = components["speech_analyzer"]
    
    # Get all persons with transcripts
    all_transcripts = {}
    for person_name, summary in {p: speech_analyzer.get_conversation_summary(p) 
                                for p in speech_processor.score_history.keys()}.items():
        if summary["utterance_count"] > 0:
            all_transcripts[person_name] = summary
    
    if not all_transcripts:
        st.info("No transcripts available yet")
        return
        
    # Create tabs for each person
    tabs = st.tabs(list(all_transcripts.keys()))
    
    for i, (person_name, tab) in enumerate(zip(all_transcripts.keys(), tabs)):
        with tab:
            summary = all_transcripts[person_name]
            utterances = summary["utterances"]
            
            if not utterances:
                st.info(f"No transcript available for {person_name}")
                continue
                
            # Display transcript with highlighted misalignment
            if config.ui.show_transcript:
                st.subheader("Transcript")
                
                # Get misalignment segments
                misalignment_segments = speech_processor.get_misalignment_segments(person_name)
                
                # Display most recent utterances first
                for utterance in reversed(utterances[-10:]):  # Show last 10 utterances
                    text = utterance.get("text", "")
                    score = utterance.get("score", 0)
                    timestamp = utterance.get("timestamp", 0)
                    
                    if text:
                        # Format timestamp
                        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                        
                        # Determine if this utterance has high misalignment
                        is_misaligned = score is not None and score >= 30
                        
                        # Display with appropriate highlighting
                        if is_misaligned:
                            st.markdown(
                                highlight_misalignment_in_text(
                                    f"**{time_str} - {person_name}:** {text}", 
                                    score
                                ),
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(f"**{time_str} - {person_name}:** {text}")
                
                # Display misalignment patterns if available
                misalignment_segments = speech_processor.get_misalignment_segments(person_name)
                if misalignment_segments:
                    with st.expander("View Detected Misalignment Patterns"):
                        for text, score, timestamp in misalignment_segments:
                            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                            st.markdown(
                                highlight_misalignment_in_text(
                                    f"**{time_str}:** \"{text}\" (Score: {score})", 
                                    score
                                ),
                                unsafe_allow_html=True
                            )
            

def settings_section(components: Dict):
    """
    Display the settings section of the UI.
    
    Args:
        components: Dictionary of system components
    """
    st.header("Settings")
    
    ui_settings = components["ui_settings"]
    camera_manager = components["camera_manager"]
    audio_recorder = components["audio_recorder"]
    combined_processor = components["combined_processor"]
    
    with st.expander("Capture Settings"):
        # Capture interval
        capture_interval = st.slider(
            "Capture Interval (seconds)",
            min_value=3.0,
            max_value=10.0,
            value=ui_settings.capture_interval,
            step=0.5
        )
        
        if capture_interval != ui_settings.capture_interval:
            ui_settings.capture_interval = capture_interval
            camera_manager.set_capture_interval(capture_interval)
            audio_recorder.set_recording_interval(capture_interval)
            st.success(f"Capture interval set to {capture_interval} seconds")
            
        # Energy threshold
        energy_threshold = st.slider(
            "Audio Energy Threshold",
            min_value=100,
            max_value=1000,
            value=ui_settings.energy_threshold,
            step=50
        )
        
        if energy_threshold != ui_settings.energy_threshold:
            ui_settings.energy_threshold = energy_threshold
            for device_name in audio_recorder.devices:
                audio_recorder.adjust_energy_threshold(device_name, energy_threshold)
            st.success(f"Audio energy threshold set to {energy_threshold}")
            
    with st.expander("Scoring Settings"):
        # Score weights
        st.subheader("Score Weights")
        cols = st.columns(2)
        
        with cols[0]:
            camera_weight = st.slider(
                "Visual Weight",
                min_value=0.0,
                max_value=1.0,
                value=ui_settings.camera_weight,
                step=0.1
            )
            
        with cols[1]:
            speech_weight = st.slider(
                "Speech Weight",
                min_value=0.0,
                max_value=1.0,
                value=ui_settings.speech_weight,
                step=0.1
            )
            
        if (camera_weight != ui_settings.camera_weight or 
            speech_weight != ui_settings.speech_weight):
            ui_settings.camera_weight = camera_weight
            ui_settings.speech_weight = speech_weight
            combined_processor.set_weights(camera_weight, speech_weight)
            st.success(f"Score weights updated: Visual={camera_weight:.1f}, Speech={speech_weight:.1f}")
            
        # Alert threshold
        alert_threshold = st.slider(
            "Alert Threshold",
            min_value=0,
            max_value=100,
            value=ui_settings.alert_threshold,
            step=5
        )
        
        if alert_threshold != ui_settings.alert_threshold:
            ui_settings.alert_threshold = alert_threshold
            combined_processor.set_alert_threshold(alert_threshold)
            st.success(f"Alert threshold set to {alert_threshold}")
            
    with st.expander("LLM Settings"):
        # Enable LLM analysis
        enable_llm = st.checkbox(
            "Enable AI Analysis",
            value=ui_settings.enable_llm_analysis
        )
        
        if enable_llm != ui_settings.enable_llm_analysis:
            ui_settings.enable_llm_analysis = enable_llm
            st.success(f"AI analysis {'enabled' if enable_llm else 'disabled'}")
            
    with st.expander("Data Transmission"):
        # Enable WebSocket
        enable_websocket = st.checkbox(
            "Enable WebSocket Transmission",
            value=ui_settings.enable_websocket
        )
        
        if enable_websocket != ui_settings.enable_websocket:
            ui_settings.enable_websocket = enable_websocket
            st.success(f"WebSocket transmission {'enabled' if enable_websocket else 'disabled'}")
            
        # WebSocket URL
        websocket_url = st.text_input(
            "WebSocket Server URL",
            value=ui_settings.websocket_url
        )
        
        if websocket_url != ui_settings.websocket_url:
            ui_settings.websocket_url = websocket_url
            st.success(f"WebSocket URL updated to {websocket_url}")
            
        # Manual transmission button
        websocket_client = components["websocket_client"]
        json_generator = components["json_generator"]
        combined_processor = components["combined_processor"]
        
        if st.button("Send Current Data Now"):
            try:
                # Get all latest scores
                all_scores = combined_processor.get_all_latest_scores()
                
                if all_scores:
                    # Get first person
                    person_name = list(all_scores.keys())[0]
                    score, details, timestamp = all_scores[person_name]
                    
                    # Generate JSON
                    camera_score = details.get("camera_score", 0)
                    camera_details = details.get("camera_details", {})
                    speech_score = details.get("speech_score", 0)
                    speech_details = details.get("speech_details", {})
                    text = details.get("speech_text", "")
                    
                    json_data = json_generator.generate_misalignment_json(
                        person_name,
                        camera_score, camera_details,
                        speech_score, speech_details, text,
                        score,
                        timestamp=timestamp
                    )
                    
                    # Save to file
                    filename = json_generator.generate_filename(person_name, "manual")
                    filepath = json_generator.save_json_to_file(json_data, filename)
                    
                    if filepath:
                        st.success(f"Data saved to {filepath}")
                    
                    # Send via WebSocket if enabled
                    if ui_settings.enable_websocket:
                        if websocket_client.send_misalignment_data(json_data):
                            st.success("Data sent via WebSocket")
                        else:
                            st.error("Failed to send data via WebSocket")
                else:
                    st.warning("No data available to send")
            except Exception as e:
                st.error(f"Error sending data: {str(e)}")
                
    with st.expander("System Actions"):
        # Reset baselines
        if st.button("Reset Visual Baselines"):
            misalignment_detector = components["misalignment_detector"]
            misalignment_detector.reset_baseline()
            st.success("Visual baselines reset for all persons")
            
        # Clear history
        if st.button("Clear All History"):
            combined_processor.clear_history()
            st.success("All history cleared")
            

def status_section(components: Dict):
    """
    Display the system status section of the UI.
    
    Args:
        components: Dictionary of system components
    """
    st.header("System Status")
    
    camera_manager = components["camera_manager"]
    audio_recorder = components["audio_recorder"]
    llm_client = components["llm_client"]
    websocket_client = components["websocket_client"]
    
    # Camera status
    camera_status = camera_manager.get_working_status()
    camera_ok = all(camera_status.values())
    
    # Audio status
    audio_status = audio_recorder.get_device_status()
    audio_ok = all(status["is_recording"] for status in audio_status.values())
    
    # LLM status
    llm_status = llm_client.get_model_status()
    llm_ok = llm_status["connected"] and llm_status["available"]
    
    # WebSocket status
    websocket_status = websocket_client.get_status()
    websocket_ok = websocket_status["connected"]
    
    # Display status indicators
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Cameras", "OK" if camera_ok else "Issues")
        if not camera_ok:
            st.caption(f"Working: {sum(camera_status.values())}/{len(camera_status)}")
            
    with cols[1]:
        st.metric("Audio", "OK" if audio_ok else "Issues")
        if not audio_ok:
            st.caption(f"Working: {sum(s['is_recording'] for s in audio_status.values())}/{len(audio_status)}")
            
    with cols[2]:
        st.metric("LLM", "OK" if llm_ok else "Issues")
        if not llm_ok:
            st.caption(f"Error: {llm_status['error']}")
            
    with cols[3]:
        st.metric("WebSocket", "OK" if websocket_ok else "Disconnected")
        if not websocket_ok and components["ui_settings"].enable_websocket:
            st.caption(f"Error: {websocket_status['error']}")
            
    # Detailed status (expandable)
    with st.expander("Detailed Status"):
        st.subheader("Camera Status")
        for name, status in camera_status.items():
            st.write(f"{name}: {'Working' if status else 'Not Working'}")
            
        st.subheader("Audio Status")
        for name, status in audio_status.items():
            st.write(f"{name}: {'Recording' if status['is_recording'] else 'Not Recording'}, " +
                    f"Active: {'Yes' if status['active'] else 'No'}")
            
        st.subheader("LLM Status")
        st.write(f"Connected: {'Yes' if llm_status['connected'] else 'No'}")
        st.write(f"Model: {llm_status['model']}")
        st.write(f"Available: {'Yes' if llm_status['available'] else 'No'}")
        
        st.subheader("WebSocket Status")
        st.write(f"Connected: {'Yes' if websocket_status['connected'] else 'No'}")
        st.write(f"Server: {websocket_status['server_url']}")
        st.write(f"Queue Size: {websocket_status['queue_size']}")