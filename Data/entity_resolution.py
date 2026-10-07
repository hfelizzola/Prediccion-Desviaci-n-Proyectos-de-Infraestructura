import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import jellyfish
import re
from collections import defaultdict

class SupplierEntityResolver:
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        
    def preprocess_supplier_name(self, name):
        """Limpia y normaliza nombres de proveedores"""
        if pd.isna(name):
            return ""
        
        # Convertir a string y minúsculas
        name = str(name).lower()
        
        # Remover sufijos comunes de empresas
        company_suffixes = ['inc', 'corp', 'corporation', 'ltd', 'limited', 'llc', 
                          'co', 'company', 's.a.', 'sa', 'srl', 'ltda']
        
        # Remover caracteres especiales y normalizar
        name = re.sub(r'[^\w\s]', ' ', name)
        name = ' '.join(name.split())
        
        # Remover sufijos comunes al final
        words = name.split()
        if words and words[-1] in company_suffixes:
            words = words[:-1]
            name = ' '.join(words)
        
        return name.strip()
    
    def standardize_supplier_id(self, supplier_id):
        """Estandariza supplierId para que sea solo numérico"""
        if pd.isna(supplier_id):
            return None
        
        # Convertir a string
        id_str = str(supplier_id)
        
        # Extraer solo números
        numeric_only = re.sub(r'[^0-9]', '', id_str)
        
        # Si no hay números, retornar None
        if not numeric_only:
            return None
        
        # Convertir a entero para eliminar ceros a la izquierda
        try:
            return int(numeric_only)
        except ValueError:
            return None
    
    def create_id_mapping(self, df):
        """Crea mapeo de IDs originales a IDs estandarizados"""
        print("Estandarizando supplier IDs...")
        
        # Aplicar estandarización
        df_temp = df.copy()
        df_temp['standardized_id'] = df_temp['supplierId'].apply(self.standardize_supplier_id)
        
        # Identificar IDs problemáticos
        problematic_ids = df_temp[df_temp['standardized_id'].isna()]
        if not problematic_ids.empty:
            print(f"⚠️  Advertencia: {len(problematic_ids)} supplier IDs no pudieron ser estandarizados:")
            for _, row in problematic_ids.head(5).iterrows():
                print(f"   - Original: '{row['supplierId']}' -> No contiene números")
            if len(problematic_ids) > 5:
                print(f"   ... y {len(problematic_ids) - 5} más")
        
        # Identificar duplicados en IDs estandarizados
        standardized_counts = df_temp['standardized_id'].value_counts()
        duplicate_ids = standardized_counts[standardized_counts > 1]
        
        if not duplicate_ids.empty:
            print(f"⚠️  Advertencia: {len(duplicate_ids)} IDs estandarizados aparecen múltiples veces:")
            for std_id, count in duplicate_ids.head(5).items():
                original_ids = df_temp[df_temp['standardized_id'] == std_id]['supplierId'].tolist()
                print(f"   - ID estandarizado {std_id} ({count} veces): {original_ids}")
            if len(duplicate_ids) > 5:
                print(f"   ... y {len(duplicate_ids) - 5} más")
        
        # Crear mapeo de original a estandarizado
        id_mapping = dict(zip(df_temp['supplierId'], df_temp['standardized_id']))
        
        print(f"✅ Estandarización completada:")
        print(f"   - IDs procesados: {len(df)}")
        print(f"   - IDs estandarizados exitosamente: {df_temp['standardized_id'].notna().sum()}")
        print(f"   - IDs únicos después de estandarización: {df_temp['standardized_id'].nunique()}")
        
        return id_mapping, df_temp['standardized_id']
    
    def calculate_supplier_similarity(self, name1, name2):
        """Calcula similitud específica para nombres de proveedores"""
        if not name1 or not name2:
            return 0.0
        
        similarities = {}
        
        # Jaro-Winkler (excelente para nombres)
        similarities['jaro_winkler'] = jellyfish.jaro_winkler_similarity(name1, name2)
        
        # Token sort ratio (maneja diferentes órdenes de palabras)
        similarities['token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100
        
        # Token set ratio (maneja palabras adicionales/faltantes)
        similarities['token_set'] = fuzz.token_set_ratio(name1, name2) / 100
        
        # Partial ratio (para casos donde uno es substring del otro)
        similarities['partial'] = fuzz.partial_ratio(name1, name2) / 100
        
        # Ratio simple
        similarities['ratio'] = fuzz.ratio(name1, name2) / 100
        
        # Score compuesto ponderado para nombres de empresas
        weights = {
            'token_set': 0.35,      # Más peso para manejar variaciones
            'jaro_winkler': 0.25,   # Bueno para errores tipográficos
            'token_sort': 0.20,     # Para diferentes órdenes
            'partial': 0.15,        # Para substrings
            'ratio': 0.05          # Menor peso para match exacto
        }
        
        composite_score = sum(similarities[metric] * weight 
                            for metric, weight in weights.items())
        
        return composite_score, similarities
    
    def create_supplier_blocks(self, df):
        """Crea bloques basados en características de nombres de proveedores"""
        blocks = defaultdict(list)
        
        for idx, row in df.iterrows():
            name = self.preprocess_supplier_name(row['supplierName'])
            
            if name:
                # Crear múltiples claves de bloqueo
                words = name.split()
                
                # Bloque por primera palabra
                if words:
                    first_word_key = words[0][:4]  # Primeras 4 letras
                    blocks[f"first_{first_word_key}"].append((idx, row))
                
                # Bloque por soundex de primera palabra
                if words:
                    soundex_key = jellyfish.soundex(words[0])
                    blocks[f"soundex_{soundex_key}"].append((idx, row))
                
                # Bloque por longitud aproximada del nombre
                length_bucket = len(name) // 5 * 5  # Agrupar por rangos de 5 caracteres
                blocks[f"length_{length_bucket}"].append((idx, row))
        
        return blocks
    
    def find_similar_suppliers(self, df):
        """Encuentra proveedores similares"""
        print(f"Procesando {len(df)} proveedores...")
        
        # Crear bloques para eficiencia
        blocks = self.create_supplier_blocks(df)
        
        similarity_results = []
        comparisons_made = 0
        
        # Comparar dentro de cada bloque
        for block_key, records in blocks.items():
            if len(records) < 2:
                continue
            
            for i, (idx1, row1) in enumerate(records):
                for j, (idx2, row2) in enumerate(records[i+1:], i+1):
                    name1 = self.preprocess_supplier_name(row1['supplierName'])
                    name2 = self.preprocess_supplier_name(row2['supplierName'])
                    
                    if name1 and name2 and name1 != name2:
                        composite_score, detailed_scores = self.calculate_supplier_similarity(name1, name2)
                        comparisons_made += 1
                        
                        if composite_score >= self.similarity_threshold:
                            similarity_results.append({
                                'idx1': idx1,
                                'idx2': idx2,
                                'supplier1': row1['supplierName'],
                                'supplier2': row2['supplierName'],
                                'supplierId1': row1['supplierId'],
                                'supplierId2': row2['supplierId'],
                                'similarity_score': composite_score,
                                'processed_name1': name1,
                                'processed_name2': name2,
                                **detailed_scores
                            })
        
        print(f"Comparaciones realizadas: {comparisons_made:,}")
        print(f"Pares similares encontrados: {len(similarity_results)}")
        
        return pd.DataFrame(similarity_results)
    
    def create_supplier_clusters(self, similarity_df):
        """Agrupa proveedores en clusters de entidades similares"""
        if similarity_df.empty:
            return {}
        
        # Crear grafo de similitudes
        supplier_graph = defaultdict(set)
        all_suppliers = set()
        
        for _, row in similarity_df.iterrows():
            idx1, idx2 = row['idx1'], row['idx2']
            supplier_graph[idx1].add(idx2)
            supplier_graph[idx2].add(idx1)
            all_suppliers.update([idx1, idx2])
        
        # Encontrar componentes conectados usando DFS
        visited = set()
        clusters = []
        
        def dfs(node, current_cluster):
            if node in visited:
                return
            visited.add(node)
            current_cluster.append(node)
            for neighbor in supplier_graph[node]:
                dfs(neighbor, current_cluster)
        
        for supplier_idx in all_suppliers:
            if supplier_idx not in visited:
                cluster = []
                dfs(supplier_idx, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        # Crear mapeo de índice a cluster
        idx_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters):
            for supplier_idx in cluster:
                idx_to_cluster[supplier_idx] = cluster_id
        
        print(f"Se formaron {len(clusters)} clusters de proveedores similares")
        return idx_to_cluster, clusters
    
    def unify_suppliers(self, df, choose_master='first'):
        """
        Unifica proveedores similares
        
        Args:
            df: DataFrame con columnas ['supplierName', 'supplierId']
            choose_master: 'first', 'shortest', 'longest', 'most_common'
        """
        print("="*60)
        print("INICIANDO UNIFICACIÓN DE PROVEEDORES")
        print("="*60)
        
        # PASO 1: Estandarizar supplier IDs
        id_mapping, standardized_ids = self.create_id_mapping(df)
        
        # Crear DataFrame con IDs estandarizados
        df_working = df.copy()
        df_working['standardized_supplier_id'] = standardized_ids
        
        print("\n" + "-"*40)
        
        # PASO 2: Encontrar proveedores similares
        similarity_df = self.find_similar_suppliers(df_working)
        
        if similarity_df.empty:
            print("No se encontraron proveedores similares para unificar.")
            result_df = df_working.copy()
            result_df['unified_supplier_name'] = result_df['supplierName']
            result_df['master_supplier_id'] = result_df['supplierId']
            result_df['master_standardized_id'] = result_df['standardized_supplier_id']
            result_df['original_supplier_id'] = result_df['supplierId']
            result_df['cluster_id'] = None
            return result_df, similarity_df
        
        # PASO 3: Crear clusters
        idx_to_cluster, clusters = self.create_supplier_clusters(similarity_df)
        
        # PASO 4: Crear DataFrame resultado
        result_df = df_working.copy()
        result_df['cluster_id'] = result_df.index.map(idx_to_cluster)
        result_df['unified_supplier_name'] = result_df['supplierName']
        result_df['master_supplier_id'] = result_df['supplierId']
        result_df['master_standardized_id'] = result_df['standardized_supplier_id']
        result_df['original_supplier_id'] = result_df['supplierId']
        result_df['is_duplicate'] = result_df['cluster_id'].notna()
        
        # PASO 5: Seleccionar representante maestro para cada cluster
        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_suppliers = df_working.iloc[cluster_indices]
            
            # Diferentes estrategias para elegir el maestro
            if choose_master == 'first':
                master_idx = cluster_indices[0]
            elif choose_master == 'shortest':
                master_idx = cluster_suppliers['supplierName'].str.len().idxmin()
            elif choose_master == 'longest':
                master_idx = cluster_suppliers['supplierName'].str.len().idxmax()
            elif choose_master == 'most_common':
                # Elegir el nombre más común (si hay empates, el primero)
                name_counts = cluster_suppliers['supplierName'].value_counts()
                most_common_name = name_counts.index[0]
                master_idx = cluster_suppliers[cluster_suppliers['supplierName'] == most_common_name].index[0]
            
            master_name = df_working.loc[master_idx, 'supplierName']
            master_id = df_working.loc[master_idx, 'supplierId']
            master_std_id = df_working.loc[master_idx, 'standardized_supplier_id']
            
            # Asignar datos maestros a todo el cluster
            result_df.loc[result_df['cluster_id'] == cluster_id, 'unified_supplier_name'] = master_name
            result_df.loc[result_df['cluster_id'] == cluster_id, 'master_supplier_id'] = master_id
            result_df.loc[result_df['cluster_id'] == cluster_id, 'master_standardized_id'] = master_std_id
        
        # PASO 6: Resumen de resultados
        n_duplicates = result_df['is_duplicate'].sum()
        n_clusters = len(clusters)
        n_unified = len(result_df) - n_duplicates + n_clusters
        
        print(f"\nRESUMEN DE UNIFICACIÓN:")
        print(f"- Proveedores originales: {len(df)}")
        print(f"- Proveedores duplicados encontrados: {n_duplicates}")
        print(f"- Clusters formados: {n_clusters}")
        print(f"- Proveedores después de unificación: {n_unified}")
        print(f"- Reducción: {len(df) - n_unified} proveedores ({((len(df) - n_unified)/len(df)*100):.1f}%)")
        
        return result_df, similarity_df
    
    def show_clusters_summary(self, result_df, similarity_df):
        """Muestra resumen detallado de los clusters encontrados"""
        if result_df['cluster_id'].isna().all():
            print("No hay clusters para mostrar.")
            return
        
        print("\nDETALLE DE CLUSTERS ENCONTRADOS:")
        print("="*80)
        
        for cluster_id in sorted(result_df['cluster_id'].dropna().unique()):
            cluster_data = result_df[result_df['cluster_id'] == cluster_id]
            master_name = cluster_data['unified_supplier_name'].iloc[0]
            master_std_id = cluster_data['master_standardized_id'].iloc[0]
            
            print(f"\nCLUSTER {int(cluster_id)} - Nombre maestro: '{master_name}' (ID estandarizado: {master_std_id})")
            print("-" * 50)
            
            for _, row in cluster_data.iterrows():
                status = "🎯 MAESTRO" if row['supplierName'] == row['unified_supplier_name'] else "📋 duplicado"
                std_id_text = f" -> {row['standardized_supplier_id']}" if pd.notna(row['standardized_supplier_id']) else " -> Sin números"
                print(f"  {status}: '{row['supplierName']}' (ID: {row['original_supplier_id']}{std_id_text})")
            
            # Mostrar similitudes dentro del cluster
            cluster_indices = cluster_data.index.tolist()
            cluster_similarities = similarity_df[
                (similarity_df['idx1'].isin(cluster_indices)) & 
                (similarity_df['idx2'].isin(cluster_indices))
            ]
            
            if not cluster_similarities.empty:
                avg_similarity = cluster_similarities['similarity_score'].mean()
                print(f"  Similitud promedio: {avg_similarity:.3f}")
        
        # Mostrar resumen de estandarización de IDs
        print(f"\nRESUMEN DE ESTANDARIZACIÓN DE IDs:")
        print("="*50)
        total_ids = len(result_df)
        valid_std_ids = result_df['standardized_supplier_id'].notna().sum()
        unique_std_ids = result_df['standardized_supplier_id'].nunique()
        
        print(f"- Total de supplier IDs: {total_ids}")
        print(f"- IDs estandarizados exitosamente: {valid_std_ids}")
        print(f"- IDs únicos después de estandarización: {unique_std_ids}")
        print(f"- IDs que no pudieron estandarizarse: {total_ids - valid_std_ids}")
        
        if valid_std_ids < total_ids:
            problematic = result_df[result_df['standardized_supplier_id'].isna()]
            print(f"\nIDs problemáticos (sin números):")
            for _, row in problematic.head(3).iterrows():
                print(f"  - '{row['original_supplier_id']}'")
            if len(problematic) > 3:
                print(f"  ... y {len(problematic) - 3} más")

# Función principal para usar con tu DataFrame
def unify_supplier_entities(df, similarity_threshold=0.85, master_selection='first'):
    """
    Función principal para unificar entidades de proveedores
    
    Args:
        df: DataFrame con columnas ['supplierName', 'supplierId']
        similarity_threshold: Umbral de similitud (0.0 a 1.0)
        master_selection: 'first', 'shortest', 'longest', 'most_common'
    
    Returns:
        result_df: DataFrame con proveedores unificados
        similarity_df: DataFrame con detalles de similitudes
    """
    
    # Validar entrada
    required_columns = ['supplierName', 'supplierId']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
    
    # Inicializar resolver
    resolver = SupplierEntityResolver(similarity_threshold=similarity_threshold)
    
    # Unificar proveedores
    result_df, similarity_df = resolver.unify_suppliers(df, choose_master=master_selection)
    
    # Mostrar resumen detallado
    resolver.show_clusters_summary(result_df, similarity_df)
    
    return result_df, similarity_df

# Ejemplo de uso con datos de prueba
if __name__ == "__main__":
    # Crear datos de ejemplo
    sample_data = {
        'supplierName': [
            'ABC Corporation',
            'ABC Corp',
            'ABC Corp.',
            'XYZ Limited',
            'XYZ Ltd',
            'Tech Solutions Inc',
            'Tech Solutions Incorporated',
            'Global Supply Co',
            'Global Supply Company',
            'UniqueSupplier LLC'
        ],
        'supplierId': ['S001', 'SUP-002', '003ABC', 'XYZ-004', '5', 'TECH_006', 'SOL007!', 'GS_008', '9-GLOBAL', 'UNI10@#']
    }
    
    df_example = pd.DataFrame(sample_data)
    
    print("DATOS DE EJEMPLO:")
    print(df_example)
    print()
    
    # Ejecutar unificación
    result_df, similarity_df = unify_supplier_entities(
        df_example, 
        similarity_threshold=0.8,
        master_selection='shortest'
    )
    
    print("\n" + "="*80)
    print("RESULTADO FINAL:")
    print("="*80)
    print(result_df[['supplierName', 'original_supplier_id', 'standardized_supplier_id', 'unified_supplier_name', 'master_standardized_id', 'is_duplicate']].to_string(index=False))