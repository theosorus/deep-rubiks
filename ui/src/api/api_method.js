import apiService from './api_service';


export class CubeApiMethods {

  static async getCube() {
    try {
      const response = await apiService.get('/cube/get-cube');
      return response.data;
    } catch (error) {
      console.error('Error while retrieving the cube:', error);
      throw new Error(
        error.response?.data?.message || 
        'Unable to retrieve cube state'
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

  static async shuffle(nb_moves) {
    const { data } = await apiService.get(`/cube/shuffle?nb_moves=${nb_moves}`);
    return data;
  }

}

export const cubeApi = {
  getCube: () => CubeApiMethods.getCube(),
  getMoves: () => CubeApiMethods.getMoves(),
  rotate: (move) => CubeApiMethods.rotate(move),
  reset: () => CubeApiMethods.reset(),
  shuffle: (nb_moves) => CubeApiMethods.shuffle(nb_moves),
  
  checkHealth: async () => {
    try {
      const response = await apiService.get('/health');
      return response.status === 200;
    } catch (error) {
      console.warn('API done:', error.message);
      return false;
    }
  }
};

export default CubeApiMethods;