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

}

export const cubeApi = {
  getCube: () => CubeApiMethods.getCube(),
  
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