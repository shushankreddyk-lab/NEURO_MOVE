import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import { useRef } from 'react'
import * as THREE from 'three'

const channelData = [
  { name: 'Fp1', x: -0.3, y: 0.85, z: 0.15, motor: false },
  { name: 'Fp2', x: 0.3, y: 0.85, z: 0.15, motor: false },
  { name: 'F7', x: -0.65, y: 0.55, z: 0.35, motor: false },
  { name: 'F3', x: -0.35, y: 0.55, z: 0.55, motor: true },
  { name: 'Fz', x: 0, y: 0.6, z: 0.55, motor: true },
  { name: 'F4', x: 0.35, y: 0.55, z: 0.55, motor: true },
  { name: 'F8', x: 0.65, y: 0.55, z: 0.35, motor: false },
  { name: 'FC5', x: -0.55, y: 0.3, z: 0.5, motor: true },
  { name: 'FC3', x: -0.35, y: 0.25, z: 0.7, motor: true },
  { name: 'FCz', x: 0, y: 0.25, z: 0.7, motor: true },
  { name: 'FC4', x: 0.35, y: 0.25, z: 0.7, motor: true },
  { name: 'FC6', x: 0.55, y: 0.3, z: 0.5, motor: true },
  { name: 'T7', x: -0.9, y: 0, z: 0, motor: true },
  { name: 'C5', x: -0.7, y: 0, z: 0.35, motor: true },
  { name: 'C3', x: -0.5, y: 0, z: 0.7, motor: true },
  { name: 'Cz', x: 0, y: 0, z: 0.85, motor: true },
  { name: 'C4', x: 0.5, y: 0, z: 0.7, motor: true },
  { name: 'C6', x: 0.7, y: 0, z: 0.35, motor: true },
  { name: 'T8', x: 0.9, y: 0, z: 0, motor: true },
  { name: 'CP5', x: -0.65, y: -0.25, z: 0.4, motor: true },
  { name: 'CP3', x: -0.45, y: -0.25, z: 0.65, motor: true },
  { name: 'CPz', x: 0, y: -0.25, z: 0.7, motor: true },
  { name: 'CP4', x: 0.45, y: -0.25, z: 0.65, motor: true },
  { name: 'CP6', x: 0.65, y: -0.25, z: 0.4, motor: true },
  { name: 'P7', x: -0.7, y: -0.5, z: 0.15, motor: false },
  { name: 'P3', x: -0.4, y: -0.55, z: 0.4, motor: false },
  { name: 'Pz', x: 0, y: -0.6, z: 0.4, motor: false },
  { name: 'P4', x: 0.4, y: -0.55, z: 0.4, motor: false },
  { name: 'P8', x: 0.7, y: -0.5, z: 0.15, motor: false },
  { name: 'O1', x: -0.3, y: -0.8, z: 0.15, motor: false },
  { name: 'Oz', x: 0, y: -0.85, z: 0.15, motor: false },
  { name: 'O2', x: 0.3, y: -0.8, z: 0.15, motor: false },
]

function Electrode({ channel }) {
  const meshRef = useRef()
  
  const color = channel.motor ? '#00d4ff' : '#a78bfa'
  
  return (
    <group position={[channel.x, channel.y, channel.z]}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.08}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {channel.name}
      </Text>
    </group>
  )
}

function BrainMesh() {
  return (
    <mesh rotation={[0, 0, 0]}>
      <sphereGeometry args={[0.9, 32, 32]} />
      <meshStandardMaterial
        color="#1e1b4b"
        transparent
        opacity={0.15}
        wireframe
      />
    </mesh>
  )
}

export default function EEG3DVisualization() {
  return (
    <div style={{
      width: '100%',
      height: '600px',
      background: 'linear-gradient(to bottom, #0f172a, #020617)',
      borderRadius: '12px',
      overflow: 'hidden'
    }}>
      <Canvas camera={{ position: [2, 1, 2], fov: 60 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <BrainMesh />
        {channelData.map((channel, index) => (
          <Electrode key={index} channel={channel} />
        ))}
        <OrbitControls enableZoom={true} enablePan={true} />
      </Canvas>
    </div>
  )
}
