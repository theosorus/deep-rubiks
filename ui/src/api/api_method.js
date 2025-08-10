import apiService from './api_service';


export class CubeApiMethods {

  static async getCube() {
    try {
      const response = await apiService.get('/cube/get-cube');
      return response.data;
    } catch (error) {
      console.error('Erreur lors de la récupération du cube:', error);
      throw new Error(
        error.response?.data?.message || 
        'Impossible de récupérer l\'état du cube'
      );
    }
  }

  static async getMoves() {
    const { data } = await apiService.get('/cube/moves');
    return data; 
  }

  static async rotate(move) {
    const { data } = await apiService.post('/cube/rotate', { move });
    return data;
  }

  static async reset() {
    const { data } = await apiService.get('/cube/reset');
    return data;
  }

}

export const cubeApi = {
  getCube: () => CubeApiMethods.getCube(),
  getMoves: () => CubeApiMethods.getMoves(),
  rotate: (move) => CubeApiMethods.rotate(move),
  reset: () => CubeApiMethods.reset(),
  
  checkHealth: async () => {
    try {
      const response = await apiService.get('/health');
      return response.status === 200;
    } catch (error) {
      console.warn('API non disponible:', error.message);
      return false;
    }
  }
};

export default CubeApiMethods;