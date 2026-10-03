import re

file_path = r'd:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:576]
new_code = """                            
                            for model_arch, model_name, predictions, latency in results:
                                counts = collections.Counter(predictions)
                                st.markdown("---")
                                st.subheader(f"Inference & Accuracy Verification ({model_arch})")
                                st.caption(f"Loaded Model: {model_name}")
                                
                                most_common_pred = counts.most_common(1)[0][0]
                                predicted_intent_str = class_labels.get(int(most_common_pred), str(most_common_pred))
                                
                                match = re.search(r'R(\\d+)', inf_file.name, re.IGNORECASE)
                                if match:
                                    run_num = int(match.group(1))
                                    target_group = map_run_and_marker_to_group(run_num, '', target_code)
                                    expected_label = target_group - 1
                                    ground_truth_str = class_labels.get(expected_label, f"Unknown ({expected_label})")
                                    is_match = (int(most_common_pred) == expected_label)
                                    match_text = "MATCH ✅" if is_match else "MISMATCH ❌"
                                    match_color = "green" if is_match else "red"
                                else:
                                    expected_label = -1
                                    ground_truth_str = target_code + " (Unknown Run)"
                                    is_match = False
                                    match_text = "UNKNOWN ⚠️"
                                    match_color = "orange"
                                
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("Predicted Intent", str(most_common_pred))
                                    st.caption(predicted_intent_str)
                                with col2:
                                    st.metric("Actual Intent (Ground Truth)", str(expected_label) if expected_label != -1 else "?")
                                    st.caption(ground_truth_str)
                                with col3:
                                    st.markdown(f"**Match Status**<br><h3 style='color: {match_color}; margin-top: 0;'>{match_text}</h3>", unsafe_allow_html=True)
                                with col4:
                                    st.metric("Measured True CPU Latency", f"{latency:.2f} ms")
                                with col5:
                                    confidence = 100.0 * (counts[most_common_pred] / len(predictions))
                                    st.metric("Aggregate Consistency", f"{confidence:.1f}%")
                                    
                                with st.expander(f"Show detailed trial-by-trial predictions for {model_arch}"):
                                    df = pd.DataFrame({'Trial': range(1, len(predictions) + 1), 'Prediction': predictions})
                                    st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.error(f'Error during prediction: {e}')
"""
new_lines.extend([new_code])
new_lines.extend(lines[587:])

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Cleaned up orphaned code properly.")
