import { Canvas } from '@react-three/fiber'
import { useGLTF, OrbitControls } from '@react-three/drei'
import { Suspense, useState, type ChangeEvent } from 'react'


function Model ({url}:{url: string}){
  const gltf = useGLTF(url)
  return (<primitive object={gltf.scene} scale={80}/>)
}

function App() {

  const [file, setFile] = useState<File>(null);
  const [modelUrl, setModelURL] = useState<string>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  }

  const handleUpload = () => {
    if (file) {
      const url = URL.createObjectURL(file)
      setModelURL(url)
    }
  }

  return (
    <>
      <input type="file" accept=".gltf, .glb" onChange={handleFileChange}/>

      <button disabled={file === null} onClick={handleUpload}>
        Upload File
      </button>

      {modelUrl && <Suspense>
        <Canvas>
          <ambientLight/>
          <pointLight position={[1, 1, 0]} intensity={10} />
          <Model url={modelUrl}/>
          <OrbitControls enablePan={true}/>
        </Canvas>
      </Suspense>}
    </>
  )
}

export default App
